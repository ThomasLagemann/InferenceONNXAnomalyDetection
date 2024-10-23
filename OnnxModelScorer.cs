using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;
using Microsoft.ML.OnnxRuntime; // Neu hinzugefügt
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ObjectDetection
{
    class OnnxModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mlContext;
        private InferenceSession session; // Neu hinzugefügt

        private IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();

        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.mlContext = mlContext;

            var sessionOptions = new SessionOptions();

            try
            {
                sessionOptions.AppendExecutionProvider_CUDA(0); // Verwende die erste GPU (Index 0)
                Console.WriteLine("CUDA Execution Provider wurde erfolgreich hinzugefügt.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Fehler beim Hinzufügen des CUDA Execution Providers: {ex.Message}");
                Console.WriteLine("Stelle sicher, dass die GPU-Version der ONNX Runtime und die korrekten CUDA-Bibliotheken installiert sind.");
            }

            // **Hier die InferenceSession initialisieren**
            try
            {
                session = new InferenceSession(modelLocation, sessionOptions);
                Console.WriteLine("InferenceSession wurde erfolgreich erstellt.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Fehler beim Erstellen der InferenceSession: {ex.Message}");
            }
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 416;
            public const int imageWidth = 416;
        }

        public struct TinyYoloModelSettings
        {
            public const string ModelInput = "image";
            public const string ModelOutput = "grid";
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData)
        {
            Console.WriteLine($"Images location: {imagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");

            // Bilder verarbeiten und Tensor erstellen
            var images = mlContext.Data.CreateEnumerable<ImageNetData>(testData, reuseRowObject: true).ToList();
            foreach (var image in images)
            {
                // Bild vorbereiten (z.B. Größe ändern, Pixel extrahieren)
                var bitmap = new System.Drawing.Bitmap(image.ImagePath);
                var resized = new System.Drawing.Bitmap(bitmap, ImageNetSettings.imageWidth, ImageNetSettings.imageHeight);

                // Tensor erstellen
                var input = new DenseTensor<float>(new[] { 1, 3, ImageNetSettings.imageHeight, ImageNetSettings.imageWidth });
                for (int y = 0; y < ImageNetSettings.imageHeight; y++)
                {
                    for (int x = 0; x < ImageNetSettings.imageWidth; x++)
                    {
                        var pixel = resized.GetPixel(x, y);
                        input[0, 0, y, x] = pixel.R / 255.0f;
                        input[0, 1, y, x] = pixel.G / 255.0f;
                        input[0, 2, y, x] = pixel.B / 255.0f;
                    }
                }

                // Eingabedaten für das Modell erstellen
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(TinyYoloModelSettings.ModelInput, input)
                };

                // **Prüfen, ob die Session korrekt initialisiert ist**
                if (session == null)
                {
                    throw new InvalidOperationException("InferenceSession wurde nicht korrekt initialisiert.");
                }

                // Inferenz durchführen
                using var results = session.Run(inputs);
                var output = results.FirstOrDefault(r => r.Name == TinyYoloModelSettings.ModelOutput).AsTensor<float>();

                yield return output.ToArray();
            }
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            return PredictDataUsingModel(data);
        }
    }
}
