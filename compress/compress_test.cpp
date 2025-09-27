#include <iostream>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include "../../CAESAR_C/CAESAR/dataset/dataset.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
// Include your dataset preprocessor header
// #include "scientific_dataset.h"  // Uncomment and adjust path as needed

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
    
    // Configure the scientific dataset preprocessor
    DatasetConfig config;
    config.binary_path = "tensor_data_1.bin";  // String path to binary file
    config.n_frame = 16;
    config.dataset_name = "Scientific Inference Dataset";
    config.variable_idx = 0;  // Select first variable
    config.section_range = {0, 10};  // Use sections 0-99
    config.frame_range = {0, 10};   // Use frames 0-999
    config.train_mode = false;  // Set to false for inference/test mode
    config.inst_norm = true;
    config.norm_type = "mean_range";
    
    try {
        // Load dataset using your preprocessor
        ScientificDataset dataset(config);
        std::cout << "Dataset loaded successfully!" << std::endl;
        std::cout << "Dataset size: " << dataset.size() << std::endl;
        
        // Get a sample from the dataset
        auto sample = dataset.get_item(0);  // Get first sample
        torch::Tensor input_tensor = sample["input"].to(device);
        
        std::cout << "Input tensor loaded via preprocessor. Shape: " << input_tensor.sizes() << std::endl;
        
        // Load and run the model
        std::vector<torch::Tensor> outputs;
        bool using_aoti = false;
        torch::inductor::AOTIModelPackageLoader* loader = nullptr;
        torch::jit::script::Module model;
        
        try {
            // Try loading as AOT Inductor model
            loader = new torch::inductor::AOTIModelPackageLoader("/home/eklasky/model.pt2");
            std::vector<torch::Tensor> inputs = {input_tensor};
            outputs = loader->run(inputs);
            using_aoti = true;
            std::cout << "Using AOT Inductor model" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "AOT Inductor failed, trying TorchScript: " << e.what() << std::endl;
            // Fallback to TorchScript
            model = torch::jit::load("/home/eklasky/model.pt");  // Note: .pt not .pt2
            std::vector<torch::jit::IValue> inputs = {input_tensor};
            auto result = model.forward(inputs);
            
            // Handle different output formats
            if (result.isTuple()) {
                auto tuple_result = result.toTuple();
                for (const auto& elem : tuple_result->elements()) {
                    outputs.push_back(elem.toTensor());
                }
            } else {
                outputs.push_back(result.toTensor());
            }
            using_aoti = false;
        }
        
        if (outputs.size() != 2) {
            std::cerr << "Error: Expected 2 output tensors, but got " << outputs.size() << std::endl;
            if (loader) delete loader;
            return 1;
        }
        
        torch::Tensor output1 = outputs[0];
        torch::Tensor output2 = outputs[1];
        
        std::cout << "Inference successful. Received 2 output tensors." << std::endl;
        std::cout << "Output 1 shape: " << output1.sizes() << std::endl;
        std::cout << "Output 2 shape: " << output2.sizes() << std::endl;
        
        // Save outputs to binary files
        save_tensor_to_bin(output1, "output1.bin");
        save_tensor_to_bin(output2, "output2.bin");
        
        // Optional: Process multiple samples from the dataset
        std::cout << "\nProcessing additional samples..." << std::endl;
        int num_samples_to_process = std::min(5, static_cast<int>(dataset.size()));
        
        for (int i = 1; i < num_samples_to_process; ++i) {
            auto current_sample = dataset.get_item(i);
            torch::Tensor current_input = current_sample["input"].to(device);
            
            std::vector<torch::Tensor> current_outputs;
            
            if (using_aoti && loader) {
                std::vector<torch::Tensor> current_inputs = {current_input};
                current_outputs = loader->run(current_inputs);
            } else {
                std::vector<torch::jit::IValue> current_inputs = {current_input};
                auto result = model.forward(current_inputs);
                
                if (result.isTuple()) {
                    auto tuple_result = result.toTuple();
                    for (const auto& elem : tuple_result->elements()) {
                        current_outputs.push_back(elem.toTensor());
                    }
                } else {
                    current_outputs.push_back(result.toTensor());
                }
            }
            
            std::cout << "Sample " << i << " processed. Output shapes: " 
                      << current_outputs[0].sizes() << ", " 
                      << current_outputs[1].sizes() << std::endl;
            
            // Save additional outputs with sample index
            save_tensor_to_bin(current_outputs[0], "output1_sample_" + std::to_string(i) + ".bin");
            save_tensor_to_bin(current_outputs[1], "output2_sample_" + std::to_string(i) + ".bin");
        }
        
        // Clean up
        if (loader) {
            delete loader;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
