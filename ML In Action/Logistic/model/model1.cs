using Microsoft.ML.Data;

namespace Logistic.model
{
    public class model1
    {
        [ColumnName("X1"), LoadColumn(0)]
        public float X1 { get; set; }
        [ColumnName("X2"), LoadColumn(1)]
        public float X2 { get; set; }
        [ColumnName("Class"), LoadColumn(2)]
        public int Class { get; set; }
    }
    public class ModelResult:model1
    {
        public new float X1 { get; set; }
        public new float X2 { get; set; }
        public bool Lable { get; set; }
    }

    public class Predicate
    {
        [ColumnName("PredictedLabel")]
        public bool PreLable { get; set; }
    }
}