using Microsoft.ML;
using Microsoft.ML.Data;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;
using System.Globalization;
using Microsoft.ML.Transforms;
using Blankstahlscanner_Inferenz.DataStructures;

namespace Blankstahlscanner
{

    public class ReshapeTransformer : CustomMappingFactory<ReshapeTransformerInput, ReshapeTransformerOutput>
    {
        // This is the custom mapping. We now separate it into a method, so that we can use it both in training and in loading.
        public static void Mapping(ReshapeTransformerInput input, ReshapeTransformerOutput output)
        {
            // Extrahiere die Pixelwerte als Array im HWC-Format
            var values = input.Reshape.GetValues().ToArray();
            
            int height = 256;
            int width = 256;
            int channels = 3;

            // Neues Array für die umgeordneten Pixelwerte im CHW-Format
            var reshapedValues = new float[values.Length];

            // Umordnung von HWC zu CHW
            for (int c = 0; c < channels; c++) // Für jeden Kanal
            {
                for (int h = 0; h < height; h++) // Für jede Höhe
                {
                    for (int w = 0; w < width; w++) // Für jede Breite
                    {
                        int chwIndex = c * height * width + h * width + w; // Zielindex (CHW)
                        int hwcIndex = h * width * channels + w * channels + c; // Quellindex (HWC)
                        reshapedValues[chwIndex] = values[hwcIndex];
                    }
                }
            }

            // Optional: Normalisierung von [0, 255] auf [0, 1]
            for (int i = 0; i < reshapedValues.Length; i++)
            {
                reshapedValues[i] /= 255f;
            }

            // Setze die umgeordneten Pixelwerte in das Ausgabe-VBuffer
            output.Reshape = new VBuffer<float>(reshapedValues.Length, reshapedValues);
        }

        // This factory method will be called when loading the model to get the mapping operation.
        public override Action<ReshapeTransformerInput, ReshapeTransformerOutput> GetMapping()
        {
            return Mapping;
        }
    }
    public class ReshapeTransformerOutput
    {
        [ColumnName("input")]
        [VectorType(3, 256, 256)]
        public VBuffer<float> Reshape;
    }
    public class ReshapeTransformerInput
    {
        [ColumnName("input")]
        [VectorType( 256, 256,3)]
        public VBuffer<float> Reshape;
    }




    public class AnomalyDetectionOnnx 
    {
        //decision boundary wehther to classify a candidate as true or false based on the predicted propability
        private const double predictionThreshold = 0.75d;
        private readonly MLContext mlContext;
        private readonly ITransformer model;
        public  int WindowSize => 100;

        public struct ImageNetSettings
        {
            public const int imageHeight = 256;
            public const int imageWidth = 256;
            public const float mean = 0;         //offsetImage
            public const float scale = 1;         //offsetImage
            public const bool interleavePixelColors = true; //interleavePixelColors
        }

        public struct ModelSettings
        {
            // input tensor name
            public const string ModelInput = "input";
            // output tensor name
            public const string ModelOutput = "anomaly_map";
        }

        public AnomalyDetectionOnnx(string modelLocation)
        {
            mlContext = new MLContext();
            model = LoadModel(modelLocation);
        }

        public ITransformer LoadModel(string modelLocation)
        {
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
            var pipeline = mlContext.Transforms.ResizeImages(
                outputColumnName: ModelSettings.ModelInput,
                imageWidth: ImageNetSettings.imageWidth,
                imageHeight: ImageNetSettings.imageHeight,
                inputColumnName: nameof(ImageNetData.Image))
            .Append(
                mlContext.Transforms.ExtractPixels(
                    outputColumnName: ModelSettings.ModelInput,
                    offsetImage: ImageNetSettings.mean,
                    interleavePixelColors: ImageNetSettings.interleavePixelColors,
                    outputAsFloatArray: true, 
                    scaleImage: ImageNetSettings.scale,
                    orderOfExtraction: ColorsOrder.ABGR))
            .Append(mlContext.Transforms.CustomMapping<ReshapeTransformerInput, ReshapeTransformerOutput>(
                                          (input, output) => ReshapeTransformer.Mapping(input, output),
                                          contractName: nameof(ReshapeTransformer)))
            .Append(
                mlContext.Transforms.ApplyOnnxModel(
                    modelFile: modelLocation,
                    outputColumnNames: new[] { ModelSettings.ModelOutput },
                    inputColumnNames: new[] { ModelSettings.ModelInput },
                gpuDeviceId: 0, fallbackToCpu: false)
    
            );
            var model = pipeline.Fit(data);
            return model;
        }


        public static float Sigmoid(float value)
        {
            float k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }
        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            //Stopwatch sw = Stopwatch.StartNew();
            IDataView scoredData = model.Transform(testData);
            //sw.Stop();
            //Console.WriteLine("model.Transform took " + sw.Elapsed.TotalMilliseconds + " ms");
            //DebugTransformedData(scoredData, ModelSettings.ModelInput);

            //var schema = scoredData.Schema;

            // Hole die Anomaly Map (256x256) als float[]
            //sw.Restart();
            IEnumerable<float[]> anomalyMaps = scoredData.GetColumn<float[]>(ModelSettings.ModelOutput);
            //sw.Stop();
            //Console.WriteLine("scoredData.GetColumn took " + sw.Elapsed.TotalMilliseconds + " ms");
            //DebugOnnxOutput(anomalyMaps);


            return anomalyMaps;
        }

        private void DebugOnnxOutput(IEnumerable<float[]> anomalyMaps, int maxSamples = 5)
        {
            int index = 0;

            foreach (var map in anomalyMaps.Take(maxSamples))
            {
                Console.WriteLine($"Anomaly Map #{index}: Length = {map.Length}");
                Console.WriteLine("Sample values (first 10):");
                foreach (var value in map.Skip(3596).Take(100))
                {
                    Console.WriteLine(value);
                }

                // Optional: Speichere die Map in eine CSV-Datei
                var partialMap = map.Skip(3598).Take(100);
                File.WriteAllLines($"onnx_output_debug_{index}.csv",
                    partialMap.Select(v => v.ToString(CultureInfo.InvariantCulture)));
                index++;
            }
        }
        

   

        public void DebugPreprocessedData( IEnumerable<ImageNetData> images)
        {
            var pipeline = mlContext.Transforms.ResizeImages(
                outputColumnName: ModelSettings.ModelInput,
                imageWidth: ImageNetSettings.imageWidth,
                imageHeight: ImageNetSettings.imageHeight,
                inputColumnName: nameof(ImageNetData.Image))
            .Append(
                mlContext.Transforms.ExtractPixels(
                    outputColumnName: ModelSettings.ModelInput,
                    offsetImage: ImageNetSettings.mean,
                    interleavePixelColors: ImageNetSettings.interleavePixelColors,
                    outputAsFloatArray: true,
                    scaleImage: ImageNetSettings.scale,
                    orderOfExtraction: ColorsOrder.ABGR))
            .Append(mlContext.Transforms.CustomMapping<ReshapeTransformerInput, ReshapeTransformerOutput>(
                (input, output) => ReshapeTransformer.Mapping(input, output),
                contractName: nameof(ReshapeTransformer)));

            // Führe die Pipeline aus, ohne das ONNX-Modell hinzuzufügen
            IDataView preprocessedData = pipeline.Fit(mlContext.Data.LoadFromEnumerable(images)).Transform(mlContext.Data.LoadFromEnumerable(images));

            // Debugge die Daten in der Spalte "ExtractedPixels"
            using var cursor = preprocessedData.GetRowCursor(preprocessedData.Schema.Where(col => col.Name == ModelSettings.ModelInput));
            var getter = cursor.GetGetter<VBuffer<float>>(preprocessedData.Schema[ModelSettings.ModelInput]);
            VBuffer<float> buffer = default;

            if (cursor.MoveNext())
            {
                getter(ref buffer);
                float[] values = buffer.DenseValues().ToArray();

                // Speichere oder prüfe die Werte
                Console.WriteLine($"Preprocessed data length: {values.Length}");
                Console.WriteLine("Sample values:");
                foreach (var value in values.Take(10))
                {
                    Console.WriteLine(value);
                }

                // Optional: Speichere die Daten in eine Datei
                File.WriteAllLines("preprocessed_data_debug.csv", values.Select(v => v.ToString("E17", System.Globalization.CultureInfo.InvariantCulture)));
            }
        }



        private void DebugTransformedData(IDataView inputData, string columnName)
        {
            // Erstelle den Cursor für die gewünschte Spalte
            using var cursor = inputData.GetRowCursor(inputData.Schema.Where(col => col.Name == columnName));
            var getter = cursor.GetGetter<VBuffer<float>>(inputData.Schema[columnName]);
            VBuffer<float> buffer = default;

            if (cursor.MoveNext())
            {
                // Extrahiere die Daten in den VBuffer
                getter(ref buffer);

                // Konvertiere den VBuffer in ein float[]
                float[] values = buffer.DenseValues().ToArray();
                

                // Filtere jeden dritten Wert (z. B. Kanal 0)
                var singleChannelValues = values
                    .Where((_, index) => index % 3 == 0) // Nimm nur jeden dritten Wert
                    .ToArray();

                Console.WriteLine($"Single channel data length: {values.Length}");
                Console.WriteLine("Sample values (first 10):");
                foreach (var value in singleChannelValues.Take(10))
                {
                    Console.WriteLine(value);
                }

                File.WriteAllLines("dotnet_preprocessed_input.csv",
                values.Select(v => v.ToString("E17", System.Globalization.CultureInfo.InvariantCulture)));


            }
            else
            {
                Console.WriteLine("No data found in the specified column.");
            }
        }



            

        public IEnumerable<Mat> Score(List<Mat> images)
        {

                        
            IEnumerable<ImageNetData> imageNetData = ImageNetData.ReadFromMatList(images);
            //DebugPreprocessedData(imageNetData);
            //Stopwatch sw = Stopwatch.StartNew();
            IDataView imageDataView = mlContext.Data.LoadFromEnumerable(imageNetData);
            //sw.Stop();
            //Console.WriteLine("LoadFromEnumerable took " + sw.Elapsed.TotalMilliseconds + " ms");
            //sw.Restart();
            var anomalyMaps = PredictDataUsingModel(imageDataView, model);
            //sw.Stop();
            //Console.WriteLine("PredictDataUsingModel took " + sw.Elapsed.TotalMilliseconds + " ms");
            //sw.Restart();
            var listMat = ConvertAnomalyMapsToMats(anomalyMaps);
            //sw.Stop();
            //Console.WriteLine("ConvertAnomalyMapsToMats took " + sw.Elapsed.TotalMilliseconds + " ms");
            return listMat;

            
            //var predictions_bool = predictions.Select(p => p >= predictionThreshold).ToList();
            //for (int i = 0; i < predictions.Count(); i++) 
            //{
            //    yield return (predictions_bool[i], predictions[i]);
            //}
        }

        public IEnumerable<Mat> ConvertAnomalyMapsToMats(IEnumerable<float[]> anomalyMaps, int width = 256, int height = 256)
        {
            foreach (var anomalyMap in anomalyMaps)
            {
                // Konvertiere das Float-Array in ein Mat
                Mat mat = ConvertFloatArrayToMat(anomalyMap, width, height);

                // Normalisiere und konvertiere in ein 8-Bit-Bild
                Mat mat8Bit = NormalizeAndConvertTo8Bit(mat);

                yield return mat8Bit;
            }
        }
        public Mat ConvertFloatArrayToMat(float[] data, int width, int height)
        {
            if (data.Length != width * height)
            {
                throw new InvalidOperationException("Anomaly Map dimensions do not match the expected size.");
            }

            // Erstelle eine neue OpenCv-Matrix
            Mat mat = new Mat(height, width, MatType.CV_32F);

            // Kopiere die Daten in die Matrix
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    mat.Set(y, x, data[y * width + x]);
                }
            }

            return mat;
        }

        public Mat NormalizeAndConvertTo8Bit(Mat mat)
        {
            // Normalisiere die Matrix (Float-Werte in den Bereich 0-255)
            Mat normalizedMat = new Mat();
            Cv2.Normalize(mat, normalizedMat, 0, 255, NormTypes.MinMax);
           
            // Konvertiere die Matrix in ein 8-Bit-Graustufenbild
            Mat mat8Bit = new Mat();
            normalizedMat.ConvertTo(mat8Bit, MatType.CV_8U);
         

            return mat8Bit;
        }







    }
}
