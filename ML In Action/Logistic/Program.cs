using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Logistic.model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using PLplot;

namespace Logistic
{
    class Program
    {
        #region Path Helper
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"../../../dataset";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/testSet.txt";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/testSet.txt";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);
        private static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
        #endregion
        private double Sigmoid(double x)
        {
            return (double)1 / (1 + Math.Exp(-1 * x));
        }

        private model1[] InitDataSet()
        {
            model1[] ap = new model1[100];
            StreamReader sR = new StreamReader(TrainDataPath, Encoding.UTF8);
            int i = 0;
            string nextLine;
            while ((nextLine = sR.ReadLine()) != null)
            {
                string[] data = nextLine.Split('\t');
                ap[i] = new model1()
                {
                    X1 = float.Parse(data[0]),
                    X2 = float.Parse(data[1]),
                    Class = int.Parse(data[2])
                };
                i++;
            }
            sR.Close();
            return ap;
        }

        private void MLNETTrain(MLContext mlc)
        {
            IDataView trainData = mlc.Data.LoadFromTextFile<model1>(TrainDataPath);
            var b = mlc.Transforms.Conversion.ConvertType(new[]
            {
                new InputOutputColumnPair("Lable","Class")
            }, DataKind.Boolean);
            var transformer = b.Fit(trainData);
            var transformedData = transformer.Transform(trainData);
            var convertedData = mlc.Data.CreateEnumerable<ModelResult>(transformedData, true);
            trainData = mlc.Data.LoadFromEnumerable<ModelResult>(convertedData);
            var a = mlc.Transforms.Concatenate("Features", new[] { "X1", "X2" });
            var options = new LbfgsLogisticRegressionBinaryTrainer.Options()
            {
                LabelColumnName = "Lable",
                FeatureColumnName = "Features",
                MaximumNumberOfIterations = 100,
                OptimizationTolerance = 1e-8f
            };
            var trainer = mlc.BinaryClassification.Trainers.LbfgsLogisticRegression(options);
            var trainPipe = a.Append(trainer);
            // Console.WriteLine("Starting training");
            ITransformer model = trainPipe.Fit(trainData);
            // Console.WriteLine("Training complete");
            IDataView predictions = model.Transform(trainData);
            var metrics = mlc.BinaryClassification.
              EvaluateNonCalibrated(predictions, "PredictedLabel");
            // Console.Write("Model accuracy on training data = ");
            // Console.WriteLine(metrics.Accuracy.ToString("F4") + "\n");
            var models = InitDataSet();
            var pe = mlc.Model.CreatePredictionEngine<model1, Predicate>(model);
            int tcnt = 0;
            for (int i = 0; i < models.Length; i++)
            {
                var Y = pe.Predict(models[i]);
                // Console.WriteLine("Predicted: {0},Actual:{1}"
                // ,Y.PreLable
                // ,models[i].Class==1?true:false);
                if(models[i].Class==1?true:false == Y.PreLable)
                {
                    tcnt++;
                }
            }
            Console.WriteLine("The ML.NET Predicate Correct Rate is {0}%", ((double)tcnt / models.Length) * 100);
        }


        private void Graph(string[] args,
        double[] x1, double[] y1,
        double[] x2, double[] y2,
        double[] x3, double[] y3)
        {
            string chartFileName = string.Empty;
            var pl = new PLStream();
            if (args.Length == 1 && args[0] == "svg")
            {
                chartFileName = @"Logistic.svg";
                pl.sdev("svg");
                pl.sfnam("Logistic.svg");
            }
            else
            {
                chartFileName = @"Logistic.png";
                pl.sdev("pngcairo");
                pl.sfnam("Logistic.png");
            }
            pl.spal0("cmap0_alternate.pal");
            pl.init();

            const int xMin = -4;
            const int xMax = 4;
            const int yMin = -5;
            const int yMax = 20;
            pl.env(xMin, xMax, yMin, yMax, AxesScale.Independent, AxisBox.BoxTicksLabels);
            // Set scaling for mail title text 125% size of default
            pl.lab("X1", "X2", "Title");
            pl.col0(3);
            pl.sym(x1, y1, (char)228);
            pl.col0(9);
            pl.sym(x2, y2, (char)228);
            pl.col0(2);
            pl.line(x3, y3);
            pl.eop();
            pl.gver(out var verText);
            var p = new Process();
            string chartFileNamePath = @".\" + chartFileName;
            p.StartInfo = new ProcessStartInfo(chartFileNamePath)
            {
                UseShellExecute = true
            };
            p.Start();
        }

        private double[] GradAscent(model1[] data, double alpha, double[] weight, int maxCircle)
        {
            while (maxCircle >= 0)
            {
                for (int i = 0; i < data.Length; i++)
                {
                    var d = data[i].X1 * weight[0] + data[i].X2 * weight[1] + weight[2];
                    var fx = Sigmoid(d);
                    var error = data[i].Class - fx;
                    // has error
                    weight[0] += data[i].X1 * error * alpha;
                    weight[1] += data[i].X2 * error * alpha;
                    weight[2] += 1 * error * alpha;
                }
                maxCircle--;
            }
            return weight;
        }

        public (double[], double[]) InitLine(double[] weight)
        {
            (double[], double[]) temp;
            temp.Item1 = new double[201];
            temp.Item2 = new double[201];
            int j = 0;
            for (double i = -5; i < 5; i = i + 0.05)
            {
                temp.Item1[j] = i;
                temp.Item2[j] = temp.Item1[j] * (-1 * weight[0] / weight[1]) - (weight[2] / weight[1]);
                j++;
            }
            return temp;
        }

        static void Main(string[] args)
        {
            Program p = new Program();
            var data = p.InitDataSet();
            var X1 = data.Where(p => p.Class == 0).Select(p => Convert.ToDouble(p.X1)).ToArray();
            var Y1 = data.Where(p => p.Class == 0).Select(p => Convert.ToDouble(p.X2)).ToArray();
            var X2 = data.Where(p => p.Class == 1).Select(p => Convert.ToDouble(p.X1)).ToArray();
            var Y2 = data.Where(p => p.Class == 1).Select(p => Convert.ToDouble(p.X2)).ToArray();
            // you can change the MaxCircle,alpha or first weight 
            int maxCircle = 7000;
            double alpha = 0.0005;
            double[] weight = new double[3] { 1.78, 0.34, 4 };
            weight = p.GradAscent(data, alpha, weight, maxCircle);
            int wrong = 0;
            for (int i = 0; i < data.Length; i++)
            {
                var Lable = data[i].X1 * weight[0] + data[i].X2 * weight[1] + weight[2] >= 0.5 ? 1 : 0;
                if (data[i].Class != Lable)
                {
                    wrong++;
                }
            }
            // the graph
            // var line = p.InitLine(weight);
            // p.Graph(args, X1, Y1, X2, Y2, line.Item1, line.Item2);
            Console.WriteLine("The My Alogorithm Predicate Correct Rate is {0}%", ((double)(data.Length - wrong) / data.Length) * 100);
            p.MLNETTrain(new MLContext(seed: 1));
        }
    }
}
