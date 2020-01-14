using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using KNN.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using PLplot;

namespace KNN
{
    class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"../../../dataset";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/datingTestSet.txt";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/datingTestSet.txt";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);
        private static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        private static void InitDataSet(MLContext mlContext)
        {
            IDataView data = mlContext.Data.LoadFromTextFile<Appointment>(TrainDataPath);
            var cnt = data.GetColumn<float>(nameof(Appointment.IceCream)).Count();
            Console.WriteLine(cnt);
        }
        private static Appointment[] InitDataSet()
        {
            Appointment[] ap = new Appointment[1000];
            StreamReader sR = new StreamReader(TrainDataPath, Encoding.UTF8);
            int i = 0;
            string nextLine;
            while ((nextLine = sR.ReadLine()) != null)
            {
                string[] data = nextLine.Split('\t');
                ap[i] = new Appointment()
                {
                    PlaneTime = float.Parse(data[0]),
                    GameTime = float.Parse(data[1]),
                    IceCream = float.Parse(data[2]),
                    LikeTread = data[3]
                };
                i++;
            }
            sR.Close();
            return ap;

        }
        private static void AutoNorm(Appointment[] data)
        {
            double range1 = data.Max(p => p.GameTime) - data.Min(p => p.GameTime);
            double range2 = data.Max(p => p.PlaneTime) - data.Min(p => p.PlaneTime);
            double range3 = data.Max(p => p.IceCream) - data.Min(p => p.IceCream);
            data.All(p =>
            {
                p.GameTime = (p.GameTime - data.Min(q => q.GameTime)) / range1;
                p.PlaneTime = (p.PlaneTime - data.Min(q => q.PlaneTime)) / range2;
                p.IceCream = (p.IceCream - data.Min(q => q.IceCream)) / range3;
                return true;
            });
        }
        private static void Graph(string[] args, double[] x, double[] y)
        {
            string chartFileName = string.Empty;
            // create PLplot object
            var pl = new PLStream();

            // use SVG backend and write to SineWaves.svg in current directory
            if (args.Length == 1 && args[0] == "svg")
            {
                chartFileName = @"SineWaves.svg";
                pl.sdev("svg");
                pl.sfnam("SineWaves.svg");
            }
            else
            {
                chartFileName = @"SineWaves.png";
                pl.sdev("pngcairo");
                pl.sfnam("SineWaves.png");
            }
            pl.spal0("cmap0_alternate.pal");
            pl.init();

            const int xMin = 0;
            const int xMax = 1;
            const int yMin = 0;
            const int yMax = 1;
            pl.env(xMin, xMax, yMin, yMax, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);

            // Set scaling for mail title text 125% size of default
            pl.schr(0, 0.75);
            pl.lab("X-axis", "Y-axis", "Title");
            pl.col0(3);
            pl.sym(x, y, (char)210);
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
        private static string KNN(Appointment[] data, Appointment test)
        {
            var d = data.OrderBy(p =>
                Math.Pow((p.GameTime - test.GameTime), 2) +
                Math.Pow((p.IceCream - test.IceCream), 2) +
                Math.Pow((p.PlaneTime - test.PlaneTime), 2))
                .Take(25).GroupBy(p => p.LikeTread)
                .Select(p => new
                {
                    type = p.Key,
                    cnt = p.Count()
                }).OrderByDescending(p => p.cnt);
            return d.First().type;
        }
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            var data = InitDataSet();
            AutoNorm(data);
            Console.WriteLine("Predicate: " + KNN(data, data[572]));
            Console.WriteLine("Actual: " + data[572].LikeTread);
            //Graph(args, data.Select(p => p.PlaneTime).ToArray(), data.Select(p => p.IceCream).ToArray());
        }
    }
}
