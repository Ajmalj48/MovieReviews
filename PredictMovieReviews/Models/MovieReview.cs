using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PredictMovieReviews.Models
{
    public class MovieReview
    {
        public string Text { get; set; }
        public string Sentiment { get; set; }

        [VectorType(1000000000)]
        public float[] Features { get; set; }
    }

    public class SentimentPrediction
    {
        public string Text { get; set; }
        public bool Prediction { get; set; }
    }
}

