#include "inference.h"
#include <regex>

#define benchmark
YOLO::YOLO() {

}


YOLO::~YOLO() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif


template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}


bool YOLO::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }
        if (iImg.cols >= iImg.rows)
        {
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        }
        else
        {
            resizeScales = iImg.rows / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;

    return true;
}


bool YOLO::CreateSession(DL_INIT_PARAM& iParams) {
    char* Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        Ret = "model path is error";
        std::cout << Ret << std::endl;
        return Ret;
    }
    try
    {
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        halfEnable = iParams.halfEnable;
        nms = iParams.nms;
        task = iParams.task;
        cudaEnable = iParams.cudaEnable;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable)
        {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();
        return true;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[YOLO]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        return "[YOLO]:Create session failed.";
    }
}


bool YOLO::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif // benchmark
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (!halfEnable)   // FP32
    {
        float* blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    }
    else               // FP16
    {
#ifdef USE_CUDA
        half* blob = new half[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
    }

    return true;
}


template<typename N>
bool YOLO::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
    std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif // benchmark
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif // benchmark

    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete[] blob;

    int signalResultNum = outputNodeDims[1];
    int strideNum = outputNodeDims[2];
    if ( task ==DETECT ) {
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        cv::Mat rawData;
        if (!halfEnable)
        {
            // FP32
            rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
        }
        else
        {
            // FP16
            rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }

        rawData = rawData.t();
        float* data = (float*)rawData.data;
        int num_classes = signalResultNum - 4;

        for (int i = 0; i < strideNum; ++i)
        {
            float* classesScores = data + 4;
            cv::Mat scores(1, num_classes, CV_32FC1, classesScores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > rectConfidenceThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * resizeScales);
                int top = int((y - 0.5 * h) * resizeScales);

                int width = int(w * resizeScales);
                int height = int(h * resizeScales);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
            data += signalResultNum;
        }
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
        for (int i = 0; i < nmsResult.size(); ++i)
        {
            int idx = nmsResult[i];
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            oResult.push_back(result);
        }
    }
    else if ( task == SEGMENT ) {
        Ort::Value& protoTensor = outputTensor[1];
        auto protoInfo = protoTensor.GetTensorTypeAndShapeInfo();
        auto protoDims = protoInfo.GetShape(); // [1, 32, h, w]

        auto protoOutput =
            protoTensor.GetTensorMutableData<typename std::remove_pointer<N>::type>();
            std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> mask_coeffs;

    cv::Mat rawData;
    if (!halfEnable)
        rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
    else {
        rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
        rawData.convertTo(rawData, CV_32F);
    }

    rawData = rawData.t();
    float* data = (float*)rawData.data;

    int num_classes = signalResultNum - 4 - 32;

    for (int i = 0; i < strideNum; ++i)
    {
        float* classesScores = data + 4;
        cv::Mat scores(1, num_classes, CV_32FC1, classesScores);

        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > rectConfidenceThreshold)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * resizeScales);
            int top  = int((y - 0.5 * h) * resizeScales);
            int width  = int(w * resizeScales);
            int height = int(h * resizeScales);

            boxes.emplace_back(left, top, width, height);

            // -------- mask coeff (32) --------
            std::vector<float> coeff(32);
            memcpy(coeff.data(), data + 4 + num_classes, 32 * sizeof(float));
            mask_coeffs.push_back(coeff);
        }
        data += signalResultNum;
    }

    // -------- NMS --------
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences,
                      rectConfidenceThreshold, iouThreshold, nmsResult);

        // -------- Proto reshape --------
        int proto_h = protoDims[2];
        int proto_w = protoDims[3];
        cv::Mat protoMat(32, proto_h * proto_w, CV_32F, protoOutput);

        for (int idx : nmsResult)
        {
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];

            // -------- mask = coeff * proto --------
            cv::Mat coeffMat(1, 32, CV_32F, mask_coeffs[idx].data());
            cv::Mat mask = coeffMat * protoMat; // [1, h*w]
            mask = mask.reshape(1, proto_h);   // [160, 160]

            // sigmoid
            cv::exp(-mask, mask);
            mask = 1.0 / (1.0 + mask);

            cv::resize(mask, mask, cv::Size(imgSize.at(0), imgSize.at(1)));

            // 2. 去除预处理(PreProcess)引入的 Padding (黑边)
            int valid_w = int(iImg.cols / resizeScales);
            int valid_h = int(iImg.rows / resizeScales);

            // 边界保护
            if (valid_w > mask.cols) valid_w = mask.cols;
            if (valid_h > mask.rows) valid_h = mask.rows;

            // 截取有效区域
            cv::Rect valid_roi(0, 0, valid_w, valid_h);
            cv::Mat valid_mask = mask(valid_roi);

            cv::resize(valid_mask, result.mask, iImg.size());

            // 4. 根据检测框(Bounding Box)裁剪 Mask
            // 确保 box 在图像范围内
            cv::Rect box = boxes[idx] & cv::Rect(0, 0, iImg.cols, iImg.rows);

            // 创建一个全黑的 Mask，只把 box 区域内的 Mask 拷贝进去
            cv::Mat finalMask = cv::Mat::zeros(iImg.size(), CV_32F);
            result.mask(box).copyTo(finalMask(box));

            result.mask = finalMask;

            oResult.push_back(result);
        }
    }
    else if ( task == POSE ) {

    }
    else {
        return "task error";
    }

#ifdef benchmark
        clock_t starttime_4 = clock();
        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
        else
        {
            std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
#endif // benchmark

    return true;
}

bool YOLO::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (!halfEnable)
    {
        float* blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
            YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
            outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
    }
    else
    {
#ifdef USE_CUDA
        half* blob = new half[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
#endif
    }
    return true;
}
