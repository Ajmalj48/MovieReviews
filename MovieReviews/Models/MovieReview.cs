using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovieReviews.Models
{
    public class MovieReview
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public string Sentiment { get; set; }
    }

}
