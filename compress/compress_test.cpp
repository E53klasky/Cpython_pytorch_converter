#include <iostream>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include "../../CAESAR_C/CAESAR/dataset/dataset.h"

void save_tensor_to_bin(const torch::Tensor& tensor, const std::string& filename) {
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    file.write(
        reinterpret_cast<const char*>(cpu_tensor.data_ptr<float>()),
        cpu_tensor.numel() * sizeof(float)
    );
    file.close();
    std::cout << "Tensor saved to " << filename << std::endl;
}

int main() {
    c10::InferenceMode mode;

    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Running on GPU." << std::endl;
        device = torch::kCUDA;
    } else {
        std::cout << "Running on CPU." << std::endl;
    }

    DatasetConfig config;
    config.binary_path = "tensor_data_1.bin";
    config.n_frame = 8;
    config.dataset_name = "Scientific Inference Dataset";
    config.variable_idx = 0;
    config.section_range = {0, 12};
    config.frame_range = {0, 256};
    config.train_mode = false;
    config.inst_norm = true;
    config.norm_type = "mean_range";
    config.test_size = {256, 256};
    config.n_overlap = 0;

    try {
  
        ScientificDataset dataset(config);
        std::cout << "Dataset loaded successfully! Size: " << dataset.size() << std::endl;

   
        auto sample = dataset.get_item(0);
        torch::Tensor input_tensor = sample["input"].to(device);
        std::cout << "Input tensor shape: " << input_tensor.sizes() << std::endl;

    
        torch::inductor::AOTIModelPackageLoader loader("/home/eklasky/model.pt2");
        std::vector<torch::Tensor> inputs = {input_tensor};
        std::vector<torch::Tensor> outputs = loader.run(inputs);

        if (outputs.size() != 2) {
            std::cerr << "Error: Expected 2 output tensors, got " << outputs.size() << std::endl;
            return 1;
        }

        torch::Tensor output1 = outputs[0];
        torch::Tensor output2 = outputs[1];

        std::cout << "Inference successful." << std::endl;
        std::cout << "Output 1 shape: " << output1.sizes() << std::endl;
        std::cout << "Output 2 shape: " << output2.sizes() << std::endl;

 
        size_t input_size = input_tensor.numel() * sizeof(double);
        size_t output1_size = output1.numel() * sizeof(float);
        size_t output2_size = output2.numel() * sizeof(float);
        size_t total_compressed = output1_size + output2_size;

        double compression_ratio = static_cast<double>(input_size) / total_compressed;
        double space_saved = 100.0 * (1.0 - static_cast<double>(total_compressed) / input_size);

        std::cout << "\n=== Single Sample Compression ===" << std::endl;
        std::cout << "Input size:        " << input_size << " bytes (" 
                  << input_size / 1024.0 << " KB)" << std::endl;
        std::cout << "Output1 size:      " << output1_size << " bytes (" 
                  << output1_size / 1024.0 << " KB)" << std::endl;
        std::cout << "Output2 size:      " << output2_size << " bytes (" 
                  << output2_size / 1024.0 << " KB)" << std::endl;
        std::cout << "Total compressed:  " << total_compressed << " bytes (" 
                  << total_compressed / 1024.0 << " KB)" << std::endl;
        std::cout << "Compression ratio: " << compression_ratio << ":1" << std::endl;
        std::cout << "Space saved:       " << space_saved << "%" << std::endl;

        save_tensor_to_bin(output1, "output1.bin");
        save_tensor_to_bin(output2, "output2.bin");


        std::cout << "\n=== Processing Multiple Samples ===" << std::endl;
        int num_samples = std::min(5, static_cast<int>(dataset.size()));
        
        size_t total_input_size = input_size;
        size_t total_output_size = total_compressed;

        for (int i = 1; i < num_samples; ++i) {
            auto current_sample = dataset.get_item(i);
            torch::Tensor current_input = current_sample["input"].to(device);
            std::vector<torch::Tensor> current_outputs = loader.run({current_input});
            
            size_t curr_input_size = current_input.numel() * sizeof(float);
            size_t curr_output_size = current_outputs[0].numel() * sizeof(float) + 
                                      current_outputs[1].numel() * sizeof(float);
            
            total_input_size += curr_input_size;
            total_output_size += curr_output_size;
            
            save_tensor_to_bin(current_outputs[0], "output1_sample_" + std::to_string(i) + ".bin");
            save_tensor_to_bin(current_outputs[1], "output2_sample_" + std::to_string(i) + ".bin");
            std::cout << "Sample " << i << " processed. Output shapes: "
                      << current_outputs[0].sizes() << ", " << current_outputs[1].sizes() << std::endl;
        }

        std::cout << "\n=== Total Compression Statistics (First " << num_samples << " Samples) ===" << std::endl;
        std::cout << "Total input size:      " << total_input_size << " bytes (" 
                  << total_input_size / 1024.0 << " KB, " 
                  << total_input_size / (1024.0 * 1024.0) << " MB)" << std::endl;
        std::cout << "Total compressed size: " << total_output_size << " bytes (" 
                  << total_output_size / 1024.0 << " KB, " 
                  << total_output_size / (1024.0 * 1024.0) << " MB)" << std::endl;
        std::cout << "Average compression:   " << static_cast<double>(total_input_size) / total_output_size 
                  << ":1" << std::endl;
        std::cout << "Total space saved:     " << 100.0 * (1.0 - static_cast<double>(total_output_size) / total_input_size) 
                  << "%" << std::endl;


        std::cout << "\n=== Full Dataset Projection (All " << dataset.size() << " Samples) ===" << std::endl;
        size_t projected_input = total_input_size * dataset.size() / num_samples;
        size_t projected_output = total_output_size * dataset.size() / num_samples;
        std::cout << "Projected input size:      " << projected_input / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Projected compressed size: " << projected_output / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Projected space saved:     " << projected_input / (1024.0 * 1024.0) - projected_output / (1024.0 * 1024.0) 
                  << " MB" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
