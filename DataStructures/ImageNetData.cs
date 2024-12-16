using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using OpenCvSharp;

namespace Blankstahlscanner_Inferenz.DataStructures
{
    public class ImageNetData
    {
        [LoadColumn(0)]
        [ImageType(256, 256)]
        public MLImage Image { get; set; }

        public static IEnumerable<ImageNetData> ReadFromMatList(IList<Mat> images)
        {
            foreach (var img in images)
            {
                yield return ReadFromMat(img);
            }
        }

        public static ImageNetData ReadFromMat(Mat image)
        {
            var ms = new MemoryStream(image.ToBytes());
            return new ImageNetData { Image = MLImage.CreateFromStream(ms) };
        }
    }
}