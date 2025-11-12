using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ScottPlot;

namespace mlnet_hoax_detector
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1️⃣ Lokasi dataset
            string dataPath = @"C:\Users\ZAM SOLUTION\Downloads\10222154_Salwa Nurazizah_UTS_ML\dataset\dataset_ml.csv";

            // 2️⃣ Inisialisasi MLContext
            MLContext mlContext = new MLContext();

            Console.WriteLine($"📁 Menggunakan dataset: {dataPath}");
            Console.WriteLine("\n🚀 Melatih model...");

            // 3️⃣ Membaca dataset
            IDataView dataView = mlContext.Data.LoadFromTextFile<NewsData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            // 4️⃣ Split data untuk training dan testing
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // 5️⃣ Pipeline proses text
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Text"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // 6️⃣ Training model
            var model = pipeline.Fit(split.TrainSet);

            // 7️⃣ Evaluasi model
            Console.WriteLine("🔍 Mengevaluasi model...");
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"\n📊 Akurasi Macro: {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"📊 Akurasi Micro: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"📉 Log Loss: {metrics.LogLoss:F2}");

            // 8️⃣ Uji prediksi satu kalimat
            var predictor = mlContext.Model.CreatePredictionEngine<NewsData, NewsPrediction>(model);

            var sample = new NewsData
            {
                Text = "Pemerintah mengumumkan kebijakan baru untuk mendukung petani lokal."
            };

            var prediction = predictor.Predict(sample);
            Console.WriteLine($"\n📰 Berita: {sample.Text}");
            Console.WriteLine($"📌 Prediksi Kategori: {prediction.PredictedLabel}");

            // 9️⃣ Simpan model
            string modelPath = Path.Combine("data", "model_hoax.zip");
            Directory.CreateDirectory("data");
            mlContext.Model.Save(model, split.TrainSet.Schema, modelPath);
            Console.WriteLine($"\n✅ Model tersimpan di: {Path.GetFullPath(modelPath)}");

            // 🔟 Visualisasi hasil (jumlah data tiap kategori)
            Console.WriteLine("\n📊 Membuat grafik distribusi label...");

            var labels = mlContext.Data.CreateEnumerable<NewsData>(dataView, reuseRowObject: false)
                .Select(d => d.Label)
                .GroupBy(l => l)
                .Select(g => new { Label = g.Key, Count = g.Count() })
                .ToList();

            double[] values = labels.Select(l => (double)l.Count).ToArray();
            string[] categories = labels.Select(l => l.Label).ToArray();

            var plt = new ScottPlot.Plot();
            plt.Add.Bars(values);
            plt.Axes.Title.Label.Text = "Distribusi Kategori Dataset";
            plt.Axes.Left.Label.Text = "Jumlah Data per Kategori";
            plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(
                positions: categories.Select((_, i) => (double)i).ToArray(),
                labels: categories
            );

            plt.SavePng("chart_output.png", 800, 600);
            Console.WriteLine("📈 Grafik hasil disimpan sebagai chart_output.png");
        }
    }

    // 🧩 Struktur Data Input
    public class NewsData
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }

    // 🧠 Struktur Data Output Prediksi
    public class NewsPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
