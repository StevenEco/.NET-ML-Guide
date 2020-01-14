using Microsoft.ML.Data;

namespace KNN.Model
{
    public class Appointment
    {
        [LoadColumn(0)]
        public double PlaneTime { get; set; }

        [LoadColumn(1)]
        public double GameTime { get; set; }

        [LoadColumn(2)]
        public double IceCream { get; set; }

        [LoadColumn(3)]
        public string LikeTread { get; set; }
    }
}