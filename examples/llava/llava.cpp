#include "clip.h"
#include "llava-utils.h"
#include "console.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

std::string& process_prompt(std::string& prompt);

struct llama_data_context {
    virtual void write(const void * src, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_context() = default;
};

struct llama_data_buffer_context : llama_data_context {
    uint8_t * ptr;
    size_t size_written = 0;

    llama_data_buffer_context(uint8_t * p) : ptr(p) {}

    void write(const void * src, size_t size) override {
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

const char* default_initial_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf>\n", argv[0]);
    printf("    --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf>\n");
    printf("    --image <path/to/an/image.jpg>\n");
    printf("    [-p \"describe the image in detail.\"]\n");
    std::string initial_prompt = default_initial_prompt;
    printf("    [-I \"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\\nUSER:\"]\n");
    printf("    [--temp 0.1]\n");
    printf("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

std::string& process_prompt(std::string& prompt) {
    std::string::size_type pos = 0;
    while((pos = prompt.find("\\n", pos)) != std::string::npos) {
        prompt.replace(pos, 2, "\n");
        pos += 1;
    }

    return prompt;
}

bool is_interacting;
bool is_processing;
uint32_t unhandled_ctrl_c = 0;

llama_context * ctx_llama;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (is_interacting && is_processing) {
            is_processing = false;
            unhandled_ctrl_c++;
        } else {
            console::cleanup();
            printf("\n");
            llama_print_timings(ctx_llama);
            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        show_additional_info(argc, argv);
        return 1;
    }

    if (params.mmproj.empty()) {
        gpt_print_usage(argc, argv, params);
        show_additional_info(argc, argv);
        return 1;
    }

    const char * img_path = params.image.c_str();
    const char * clip_path = params.mmproj.c_str();

    if (params.prompt.empty()) {
        params.prompt = "describe the image in detail.";
    }
    else {
        params.prompt = process_prompt(params.prompt);
    }
    if (params.llava_prompt.empty()) {
        params.llava_prompt = default_initial_prompt;
    }
    else {
        params.llava_prompt = process_prompt(params.llava_prompt);
    }
    if (params.image.empty()) {
        img_path = NULL;
    }

    if(params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    // load and preprocess the image
    clip_image_u8 img;
    clip_image_f32 img_res;

    llama_backend_init(params.numa);

    llama_model_params model_params              = llama_model_default_params();
                       model_params.n_gpu_layers = params.n_gpu_layers;
                       model_params.main_gpu     = params.main_gpu;
                       model_params.tensor_split = params.tensor_split;
                       model_params.use_mmap     = params.use_mmap;
                       model_params.use_mlock    = params.use_mlock;

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = params.n_ctx < 2048 ? 2048 : params.n_ctx; // we need a longer context size to process image embeddings
    ctx_params.n_threads       = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
    ctx_params.seed            = params.seed;

    ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // process the prompt
    // llava chat format is "<system_prompt>USER: <image_embeddings>\n<textual_prompt>\nASSISTANT:"

    int n_past;
    int initial_n_past = 0;

    const int max_tgt_len = params.n_predict < 0 ? 256 : params.n_predict;

    // generate the response

    printf("\n");
    if(params.interactive) {
        printf("Interaction mode enabled.\n");
    }
    else {
        printf("prompt: '%s'\n", params.prompt.c_str());
    }
    printf("\n");

    size_t max_size = 0;

    std::vector<uint8_t> copy_buffer;
    bool image_saved = false;

    is_interacting = params.interactive;

    std::string img_path_buffer;

    do {
        if(image_saved) {
            llama_set_state_data(ctx_llama, copy_buffer.data());
        }

        while(!image_saved) {
            if(img_path == NULL) {
                printf("\nPlease enter a valid image path: ");
                fflush(stdout);

                unhandled_ctrl_c = 0;
                console::readline(img_path_buffer, false);

                if(unhandled_ctrl_c > 0) {
                    break;
                }

                if(img_path_buffer.size() == 0) {
                    continue;
                }

                if(img_path_buffer.at(img_path_buffer.size() - 1) == '\n') {
                    img_path_buffer.erase(img_path_buffer.size() - 1);
                }

                img_path = img_path_buffer.c_str();
            }

            if (!clip_image_load_from_file(img_path, &img)) {
                fprintf(stderr, "%s: is %s really an image file?\n", __func__, img_path);

                img_path = NULL;
                continue;
            }

            if(unhandled_ctrl_c > 2) {
                printf("\nCTRL+C received while loading image.\n\n");
                img_path = NULL;
                continue;
            }

            if (!clip_image_preprocess(ctx_clip, &img, &img_res, /*pad2square =*/ true)) {
                fprintf(stderr, "%s: unable to preprocess %s\n", __func__, img_path);

                img_path = NULL;
                continue;
            }

            if(unhandled_ctrl_c > 2) {
                printf("\nCTRL+C received while processing image.\n\n");
                img_path = NULL;
                continue;
            }

            int n_img_pos  = clip_n_patches(ctx_clip);
            int n_img_embd = clip_n_mmproj_embd(ctx_clip);

            float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip));

            if (!image_embd) {
                fprintf(stderr, "Unable to allocate memory for image embeddings\n");

                return 1;
            }

            if(unhandled_ctrl_c > 2) {
                printf("\nCTRL+C received while processing image.\n\n");
                free(image_embd);
                img_path = NULL;
                continue;
            }

            const int64_t t_img_enc_start_us = ggml_time_us();
            if (!clip_image_encode(ctx_clip, params.n_threads, &img_res, image_embd)) {
                fprintf(stderr, "Unable to encode image\n");

                return 1;
            }
            const int64_t t_img_enc_end_us = ggml_time_us();

            // make sure that the correct mmproj was used, i.e., compare apples to apples
            const int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));

            if (n_img_embd != n_llama_embd) {
                printf("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_img_embd, n_llama_embd);

                llama_free(ctx_llama);
                llama_free_model(model);
                llama_backend_free();
                free(image_embd);

                return 1;
            }
            
            llama_kv_cache_tokens_rm(ctx_llama, -1, -1);
            initial_n_past = 0;
            eval_string(ctx_llama, params.llava_prompt.c_str(), params.n_batch, &initial_n_past, true);
            eval_image_embd(ctx_llama, image_embd, n_img_pos, params.n_batch, &initial_n_past);

            if(unhandled_ctrl_c > 2) {
                printf("\nCTRL+C received while generating embedding.\n\n");
                img_path = NULL;
                free(image_embd);
                continue;
            }

            unhandled_ctrl_c = 0;

            size_t ctx_size = llama_get_state_size(ctx_llama);
            if(ctx_size != max_size) {
                max_size = ctx_size;
                printf("Using copy buffer of size: %zu\n", max_size);
                fflush(stdout);

                copy_buffer.reserve(max_size);
            }

            llama_copy_state_data(ctx_llama, copy_buffer.data());
            free(image_embd);
            {
                const float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

                printf("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / n_img_pos);
            }

            image_saved = true;
        }

        if(!image_saved) {
            break;
        }

        std::string prompt;
        if(is_interacting) {
            printf("\nWaiting for prompt (enter \\image to change the image being used)\n> ");
            console::set_display(console::user_input);

            std::string buffer;
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;

                if(unhandled_ctrl_c > 2) {
                    is_interacting = false;
                    break;
                }
            } while (another_line);

            // done taking input, reset color
            console::set_display(console::reset);

            prompt = buffer;

            if(!is_interacting) { // CTRL+C
                break;
            }

            if(prompt == "\\image\n") {
                image_saved = false;
                img_path = NULL;
                continue;
            }
            
            unhandled_ctrl_c = 0;
        }
        else {
            prompt = params.prompt;
        }

        n_past = initial_n_past;
        is_processing = true;

        eval_string(ctx_llama, (prompt + "\nASSISTANT:").c_str(), params.n_batch, &n_past, false);

        for (int i = 0; i < max_tgt_len && is_processing; i++) {
            const char * tmp = sample(ctx_llama, params, &n_past);
            if (strcmp(tmp, "</s>") == 0) break;

            printf("%s", tmp);
            fflush(stdout);
        }

        if(!is_processing) {
            printf("\nCTRL+C received.\n\n");
        }

        if(unhandled_ctrl_c > 2) {
            break;
        }

        unhandled_ctrl_c = 0;
    } while(is_interacting);

    printf("\n");

    llama_print_timings(ctx_llama);

    llama_free(ctx_llama);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
