using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Logistic.model;
using PLplot;

namespace Logistic
{
    class Program
    {
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

        private double Sigmoid(double x)
        {
            return (double)1 / Math.Exp(x);
        }

        private model1[] InitDataSet()
        {
            model1[] ap = new model1[100];
            StreamReader sR = new StreamReader(TrainDataPath, Encoding.UTF8);
            int i = 0;
            string nextLine;
            while ((nextLine = sR.ReadLine()) != null)
            {
                if (i == 99)
                {
                    Console.WriteLine();
                }
                string[] data = nextLine.Split('\t');
                ap[i] = new model1()
                {
                    X1 = double.Parse(data[0]),
                    X2 = double.Parse(data[1]),
                    Lable = int.Parse(data[2])
                };
                i++;
            }
            sR.Close();
            return ap;
        }

        private void Graph(string[] args, double[] x1, double[] y1, double[] x2, double[] y2)
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
            pl.env(xMin, xMax, yMin, yMax, AxesScale.Independent, AxisBox.CustomXYBoxTicksLabels);

            // Set scaling for mail title text 125% size of default
            pl.lab("X1", "X2", "Title");
            pl.col0(3);
            pl.sym(x1, y1, (char)228);
            pl.col0(9);
            pl.sym(x2, y2, (char)228);
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


        static void Main(string[] args)
        {
            Program p = new Program();
            var data = p.InitDataSet();
            var X1 = data.Where(p=>p.Lable == 0).Select(p => p.X1).ToArray();
            var Y1 = data.Where(p=>p.Lable == 0).Select(p => p.X2).ToArray();
            var X2 = data.Where(p=>p.Lable == 1).Select(p => p.X1).ToArray();
            var Y2 = data.Where(p=>p.Lable == 1).Select(p => p.X2).ToArray();
            p.Graph(args,X1,Y1,X2,Y2);
            Console.WriteLine(p.Sigmoid(1));
        }
    }
}
