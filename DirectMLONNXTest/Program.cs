using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Transforms.Image;

// Define asset paths
var labelsPath = "Labels.txt";
var imagePath = "dog.jpeg"; 

// Initialize MLContext
var ctx = new MLContext();

// Load image into IDataView
var dv = ctx.Data.LoadFromEnumerable(new[] { new { ImagePath = imagePath } });

// Define custom transform to normalize image pixel data
var NormalizePixels = (ImageData input, ImageData output) =>
    {
        var mean = new[] { 0.485f, 0.456f, 0.406f };
        var stddev = new[] { 0.229f, 0.224f, 0.225f };

        var transformed =
            input.Pixels.Chunk(3)
                .SelectMany(pixels =>
                {
                    return pixels.Select((pixel,idx) =>
                    {
                        return idx switch
                        {
                            0 => ((pixel / 255f) - mean[0]) / stddev[0], //R
                            1 => ((pixel / 255f) - mean[1]) / stddev[1], //G
                            2 => ((pixel / 255f) - mean[2]) / stddev[2], //B
                            _ => throw new IndexOutOfRangeException("No pixel")
                        };
                    });
                });

        output.Pixels = transformed.ToArray();
    };

// Run inference on image pixel data using ResNet50 v2 ONNX model
var RunORTInference = (ImageData input, ORTPrediction output) =>
{
    var modelInput = new[]
    {
        NamedOnnxValue.CreateFromTensor("data", new DenseTensor<float>(input.Pixels, new[] { 1, 3, 224, 224 }))
    };

    // Configure ORT Session to use DirectML
    var sessionOptions = new SessionOptions();
    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
    sessionOptions.AppendExecutionProvider_DML(0);

    // Initialize ORT session
    var modelPath = "model.onnx"; //ResNet50 v2 
    using var session = new InferenceSession(modelPath, sessionOptions);
    
    // Run inference session
    using var results = session.Run(modelInput);
    
    // Set prediction
    output.Prediction = results.First().AsEnumerable<float>().ToArray();
};

// Apply Softmax function to ONNX inference results
var PostProcess = (ORTPrediction input, ORTPrediction output) =>
{
    var sum = input.Prediction.Sum(x => (float)Math.Exp(x));
    var softmax = input.Prediction.Select(x => (float)Math.Exp(x) / sum);
    output.Prediction = softmax.ToArray();
};

// Define data processing and inferencing pipeline
var pipeline =
    ctx.Transforms.LoadImages("ImagePath", null)
        .Append(ctx.Transforms.ResizeImages("ResizedImage", 224, 224, "ImagePath"))
        .Append(ctx.Transforms.ExtractPixels("Pixels", "ResizedImage"))
        .Append(ctx.Transforms.CustomMapping(NormalizePixels, contractName: null))
        .Append(ctx.Transforms.CustomMapping(RunORTInference, contractName: null))
        .Append(ctx.Transforms.CustomMapping(PostProcess, contractName: null));

// Apply pipeline to image data
var prediction =
    pipeline
        .Fit(dv)
        .Transform(dv)
        .GetColumn<float[]>("Prediction")
        .First();

// Load labels
var labels = File.ReadLines(labelsPath).ToArray();

// Get top 10 predictions and confidence values
var top10 =
    prediction
        .Select((x, i) => new { Label = labels[i], Confidence = x })
        .OrderByDescending(x => x.Confidence)
        .Take(10);

// Output predictions
Console.WriteLine("Top 10 predictions for ResNet50 v2...");
Console.WriteLine("--------------------------------------------------------------");
foreach (var t in top10)
{
    Console.WriteLine($"Label: {t.Label}, Confidence: {t.Confidence}");
}


public class ImageData
{
    [VectorType(1,3,224,224)]
    public float[] Pixels { get; set; } 
}

public class ORTPrediction
{
    [VectorType(1, 3, 224, 224)]
    public float[] Pixels { get; set; }

    [VectorType()]
    public float[] Prediction { get; set; }
}