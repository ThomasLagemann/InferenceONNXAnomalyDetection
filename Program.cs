using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using System.Drawing.Imaging;
using System.Diagnostics;
using Blankstahlscanner_Inferenz;
using OpenCvSharp;
using static Blankstahlscanner_Inferenz.AnomalyDetectionOnnx;

class Program
{
    public static int patchSize = 256;
    static void Main(string[] args)
    {
        string onnxModelPath = "D:\\Repos\\Blankstahlscanner_Inferenz\\onnxconverter\\900EfficientAd_model2.onnx";
        string imagePath = "D:\\Repos\\Blankstahlscanner_Inferenz\\onnxconverter\\test_900_900.png";
        string outputImagePath = "D:\\Repos\\Blankstahlscanner_Inferenz\\onnxconverter\\anomaly_map_output.png";

        var inputImage = Cv2.ImRead(imagePath);
        //Cv2.Resize(inputImage, inputImage, new OpenCvSharp.Size(ImageNetSettings.imageWidth, ImageNetSettings.imageHeight));

        var mlContext = new MLContext();

        var ad =new AnomalyDetectionOnnx(onnxModelPath, mlContext);
        var input = new List<Mat>() { inputImage };
        
        Stopwatch stopWatch = new Stopwatch();
        stopWatch.Start();
        var res = ad.Score(input);
        stopWatch.Stop();
        TimeSpan ts = stopWatch.Elapsed;
        Console.WriteLine("Inference took " + ts.TotalMilliseconds.ToString());
        Cv2.ImShow("Output", res[0]);
        Cv2.WaitKey();


        //// Lade das Bild und wandle es mithilfe der ML.NET-Pipeline in einen Tensor um
        //var inputTensor = ImagePreprocessor.PreprocessImageWithMlNet(imagePath, mlContext, patchSize);

        //File.WriteAllLines("dotnet_tensor.csv", inputTensor.ToArray().Select(v => v.ToString()));

        //// Inferenz mit dem ONNX-Modell
        //var anomalyMapData = PredictWithOnnxModel(onnxModelPath, inputTensor);
        //anomalyMapData = PredictWithOnnxModel(onnxModelPath, inputTensor);
        //anomalyMapData = PredictWithOnnxModel(onnxModelPath, inputTensor);

        //// Ergebnisbild speichern
        ////SaveAnomalyMap(anomalyMapData, outputImagePath);
        //ImagePreprocessor.SaveTensorAsImage(anomalyMapData,patchSize, patchSize,outputImagePath);
    }

    static float[] PredictWithOnnxModel(string onnxModelPath, Tensor<float> inputTensor)
    {
        var sessionOptions = new SessionOptions();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
        sessionOptions.EnableMemoryPattern = true;
        sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        sessionOptions.AppendExecutionProvider_CUDA(0);

        using var session = new InferenceSession(onnxModelPath, sessionOptions);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        // Modell ausführen
        Stopwatch stopWatch = new Stopwatch();
        stopWatch.Start();
        using var results = session.Run(inputs);
        stopWatch.Stop();
        TimeSpan ts = stopWatch.Elapsed;
        Console.WriteLine("Inference took " + ts.TotalMilliseconds.ToString());
        Console.ReadKey();
        var anomalyMapTensor = results.First(r => r.Name == "anomaly_map").AsTensor<float>();

        return anomalyMapTensor.ToArray();
    }

    static void SaveHeatmapAsImage(float[] heatmap, int originalWidth, int originalHeight, int targetWidth, int targetHeight, string filePath)
    {
        // 1. Normierung auf den Bereich 0-255
        float min = heatmap.Min();
        float max = heatmap.Max();
        float range = max - min;

        if (range == 0)
        {
            range = 1; // Vermeidung von Division durch 0
        }

        byte[] normalizedHeatmap = heatmap.Select(value => (byte)(255 * (value - min) / range)).ToArray();

        // 2. Array auf das Zielbildformat umwandeln
        using var originalBitmap = new Bitmap(originalWidth, originalHeight);
        for (int y = 0; y < originalHeight; y++)
        {
            for (int x = 0; x < originalWidth; x++)
            {
                byte intensity = normalizedHeatmap[y * originalWidth + x];
                originalBitmap.SetPixel(x, y, Color.FromArgb(intensity, intensity, intensity));
            }
        }

        // 3. Reskalieren des Bildes
        using var resizedBitmap = new Bitmap(targetWidth, targetHeight);
        using (Graphics graphics = Graphics.FromImage(resizedBitmap))
        {
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBilinear;
            graphics.DrawImage(originalBitmap, 0, 0, targetWidth, targetHeight);
        }

        // 4. Speichern des Bildes
        resizedBitmap.Save(filePath, ImageFormat.Png);
    }

    static void SaveAnomalyMap(float[] anomalyMapData, string outputImagePath)
    {
        int height = patchSize;
        int width = patchSize;

        float min = anomalyMapData.Min();
        float max = anomalyMapData.Max();
        byte[] normalizedData = new byte[anomalyMapData.Length];
        for (int i = 0; i < anomalyMapData.Length; i++)
        {
            normalizedData[i] = (byte)((anomalyMapData[i] - min) / (max - min) * 255);
        }

        using var bitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
        var palette = bitmap.Palette;
        for (int i = 0; i < 256; i++)
        {
            palette.Entries[i] = Color.FromArgb(i, i, i);
        }
        bitmap.Palette = palette;

        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, bitmap.PixelFormat);
        System.Runtime.InteropServices.Marshal.Copy(normalizedData, 0, bitmapData.Scan0, normalizedData.Length);
        bitmap.UnlockBits(bitmapData);

        bitmap.Save(outputImagePath, System.Drawing.Imaging.ImageFormat.Png);
        Console.WriteLine($"Anomaly map saved to {outputImagePath}");
    }
}




public class ImagePreprocessor
{
    public static string debugOutputImagePath = "D:\\Repos\\Blankstahlscanner_Inferenz\\onnxconverter\\debug_output.png";
    public static Tensor<float> PreprocessImageWithMlNet(string imagePath, MLContext mlContext, int patchSize)
    {
        var data = new List<ImageData> { new ImageData { ImagePath = imagePath } };
        var dataView = mlContext.Data.LoadFromEnumerable(data);

        var pipeline = mlContext.Transforms.LoadImages(
                outputColumnName: "Image",
                imageFolder: Path.GetDirectoryName(imagePath),
                inputColumnName: nameof(ImageData.ImagePath))
            .Append(mlContext.Transforms.ResizeImages(
                outputColumnName: "Image",
                imageWidth: patchSize,
                imageHeight: patchSize))
            .Append(mlContext.Transforms.ExtractPixels(
                outputColumnName: "Image",
                interleavePixelColors: true, // RGB-Kanalreihenfolge
                scaleImage: 1f / 255f));     // Skaliert Pixel auf [0,1]

        var transformedData = pipeline.Fit(dataView).Transform(dataView);
        var imageColumn = transformedData.GetColumn<VBuffer<float>>("Image").First();

        float[] imageData = imageColumn.DenseValues().ToArray();
        SaveTensorAsImage(imageData, patchSize, patchSize, debugOutputImagePath);
        return new DenseTensor<float>(imageData, new int[] { 1, 3, patchSize, patchSize });
    }

    public static void SaveTensorAsImage(float[] imageData, int width, int height, string outputPath)
    {
        using var bitmap = new Bitmap(width, height);
        int index = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Hole R, G, und B-Kanäle aus dem normalisierten float[]-Array
                //byte r = (byte)(imageData[index++] * 255);
                byte g = (byte)(imageData[index++] * 255);
                //byte b = (byte)(imageData[index++] * 255);

                // Setze Pixel im Bitmap
                bitmap.SetPixel(x, y, Color.FromArgb(g, g, g));
            }
        }

        bitmap.Save(outputPath, ImageFormat.Png);
        Console.WriteLine($"Image saved to {outputPath}");
    }

    public class ImageData
    {
        public string ImagePath { get; set; }
    }
}

