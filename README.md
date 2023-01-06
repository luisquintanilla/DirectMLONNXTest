# ML.NET DirectML ONNX Runtime Execution Provider Sample - Image Classification

This sample shows how to use the DirectML ONNX Runtime Execution Provider inside an ML.NET pipeline for image classification

This is a modified version of the [Image Recognition tutorial](https://onnxruntime.ai/docs/tutorials/csharp/resnet50_csharp.html) on the ONNX Runtime documentation site.

## Prerequisites

- [.NET 6 SDK or greater](https://dotnet.microsoft.com/download)
- [Microsoft.ML.OnnxRuntime.DirectML NuGet package](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)
- [Microsoft.ML.ImageAnalytics NuGet package](https://www.nuget.org/packages/Microsoft.ML.ImageAnalytics/3.0.0-preview.22621.2)

**The DirectML Onnx Runtime Execution Provider only supports Windows.**

## Project Overview

This project adds a few [CustomMapping](https://learn.microsoft.com/dotnet/api/microsoft.ml.custommappingcatalog.custommapping?view=ml-dotnet) transforms to an ML.NET pipeline which:

1. Normalize image pixel data to a range between 0-1.
1. Configure and run an ORT inference session with DirectML enabled.
1. Post-process inference session results.

## Instructions

1. Download the [ResNet50 v2 ONNX model](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v2-7.onnx) and save it in the *DirectMLONNXText* project directory. Make sure to rename the file to *model.onnx* or update the `modelPath` variable in *Program.cs*.
1. Download the [dog.jpeg image file](https://onnxruntime.ai/images/dog.jpeg) and save it in the *DirectMLONNXText* project directory. 
1. Use the dotnet CLI or Visual Studio to run your application. If successful, the resulting output should look similar to the following:

    ```text
    2023-01-06 16:55:51.7609747 [W:onnxruntime:, inference_session.cc:491 onnxruntime::InferenceSession::RegisterExecutionProvider] Having memory pattern enabled is not supported while using the DML Execution Provider. So disabling it for this session since it uses the DML Execution Provider.
    Top 10 predictions for ResNet50 v2...
    --------------------------------------------------------------
    Label: "Golden Retriever", Confidence: 0.7697107
    Label: "Kuvasz", Confidence: 0.1426687
    Label: "Otterhound", Confidence: 0.015724588
    Label: "Clumber Spaniel", Confidence: 0.009826223
    Label: "Saluki", Confidence: 0.0062108114
    Label: "Tibetan Terrier", Confidence: 0.0057344614
    Label: "Pyrenean Mountain Dog", Confidence: 0.0050382884
    Label: "Sussex Spaniel", Confidence: 0.004918177
    Label: "Labrador Retriever", Confidence: 0.0040683337
    Label: "Tibetan Mastiff", Confidence: 0.0037927858
    ```