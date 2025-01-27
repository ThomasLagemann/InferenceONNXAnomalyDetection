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
using DefectScanner;
using OpenCvSharp;
using OpenCvSharp.Extensions;

using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;

using DefectScanner.Interfaces;
using DefectScanner.Detection;

class Program
{
    public static int patchSize = 256;
    static void Main(string[] args)
    {
        string assetsFolderPath = Path.Combine(AppContext.BaseDirectory, "Assets");
        string onnxModelPath = Path.Combine(assetsFolderPath, "model", "512_256_EfficientAd_model15.onnx");
        string input = Path.Combine(assetsFolderPath, "images", "test_9000_1920.jpg");
        //var inputImage = Cv2.ImRead(input);
        Bitmap bitmap = new Bitmap(input);

        //Cv2.Resize(inputImage, inputImage, new OpenCvSharp.Size(ImageNetSettings.imageWidth, ImageNetSettings.imageHeight));
        var ds = new DefectScannerImplementation(onnxModelPath);
        ds.Detect(bitmap);
        
        //var input1 = new List<Mat>() { inputImage };
        //var input10 = new List<Mat>() { inputImage, inputImage.Clone(), inputImage.Clone(), inputImage.Clone(), inputImage.Clone(), inputImage.Clone(), inputImage.Clone(), inputImage.Clone(), inputImage.Clone(), inputImage.Clone() };
        //var input100 = new List<Mat>() { inputImage.Clone(), inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage, inputImage };
        //Mat resFirst = ad.Score(input1).First();

        //Stopwatch stopWatch1 = new Stopwatch();
        //Stopwatch stopWatch = new Stopwatch();
        //stopWatch1.Start();
        //IEnumerable<Mat>? res = null;
        //for (int i = 0; i < 10; i++)
        //{
        //    stopWatch.Restart();
        //    res = ad.Score(input1);
        //    stopWatch.Stop();
        //    TimeSpan ts = stopWatch.Elapsed;
        //    Console.WriteLine("Inference of 1 image took " + ts.TotalMilliseconds.ToString() + " ms");
        //}
        //var resList = res.ToList();
        //var myRes1 = resList[0];

        //stopWatch1.Stop();
        //Console.WriteLine("Inference of 10 timea 1 image took " + stopWatch1.Elapsed.TotalMilliseconds.ToString() + " ms");
        
        //stopWatch1.Restart();
        //for (int i = 0; i < 10; i++)
        //{
        //    stopWatch.Restart();
        //    res = ad.Score(input10);
        //    stopWatch.Stop();
        //    TimeSpan ts = stopWatch.Elapsed;
        //    Console.WriteLine("Inference of 10 images took " + ts.TotalMilliseconds.ToString() + " ms");
        //}
        //resList = res.ToList();
        //var myRes10 = resList[0];

        //stopWatch1.Stop();
        //Console.WriteLine("Inference of 10 time 10 images took " + stopWatch1.Elapsed.TotalMilliseconds.ToString() + " ms");
        

        //Cv2.ImShow("Output", res.First());
        //Cv2.WaitKey();
        return;
       
        
    }

    static float[] PredictWithOnnxModel(InferenceSession session, Tensor<float> inputTensor)
    {
       

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
        Console.WriteLine("session.Run took " + ts.TotalMilliseconds.ToString());
        
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




//public class ImagePreprocessor
//{
//    public static string debugOutputImagePath = "D:\\Repos\\Blankstahlscanner_Inferenz\\onnxconverter\\debug_output.png";
//    public static Tensor<float> PreprocessImageWithMlNet(List<Mat> images, MLContext mlContext, int patchSize)
//    {
//        IEnumerable<ImageNetData> imageNetData = ImageNetData.ReadFromMatList(images);
//        //DebugPreprocessedData(imageNetData);
//        IDataView imageDataView = mlContext.Data.LoadFromEnumerable(imageNetData);
        

//        var pipeline = mlContext.Transforms.ResizeImages(
//                outputColumnName: ModelSettings.ModelInput,
//                imageWidth: ImageNetSettings.imageWidth,
//                imageHeight: ImageNetSettings.imageHeight,
//                inputColumnName: nameof(ImageNetData.Image))
//            .Append(
//                mlContext.Transforms.ExtractPixels(
//                    outputColumnName: ModelSettings.ModelInput,
//                    offsetImage: ImageNetSettings.mean,
//                    interleavePixelColors: ImageNetSettings.interleavePixelColors,
//                    outputAsFloatArray: true,
//                    scaleImage: ImageNetSettings.scale,
//                    orderOfExtraction: ColorsOrder.ABGR))
//            .Append(mlContext.Transforms.CustomMapping<ReshapeTransformerInput, ReshapeTransformerOutput>(
//                                          (input, output) => ReshapeTransformer.Mapping(input, output),
//                                          contractName: nameof(ReshapeTransformer)));

//        var transformedData = pipeline.Fit(imageDataView).Transform(imageDataView);
        
//        var imageColumn = transformedData.GetColumn<VBuffer<float>>(ModelSettings.ModelInput).Last();

//        float[] imageData = imageColumn.DenseValues().ToArray();
//        SaveTensorAsImage(imageData, patchSize, patchSize, debugOutputImagePath);
//        return new DenseTensor<float>(imageData, new int[] { 1, 3, patchSize, patchSize });
//    }

//    public static void SaveTensorAsImage(float[] imageData, int width, int height, string outputPath)
//    {
//        using var bitmap = new Bitmap(width, height);
//        int index = 0;

//        for (int y = 0; y < height; y++)
//        {
//            for (int x = 0; x < width; x++)
//            {
//                // Hole R, G, und B-Kanäle aus dem normalisierten float[]-Array
//                //byte r = (byte)(imageData[index++] * 255);
//                byte g = (byte)(imageData[index++] * 255);
//                //byte b = (byte)(imageData[index++] * 255);

//                // Setze Pixel im Bitmap
//                bitmap.SetPixel(x, y, Color.FromArgb(g, g, g));
//            }
//        }

//        bitmap.Save(outputPath, ImageFormat.Png);
//        Console.WriteLine($"Image saved to {outputPath}");
//    }

//    public class ImageData
//    {
//        public string ImagePath { get; set; }
//    }
//}

