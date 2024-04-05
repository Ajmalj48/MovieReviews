using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using PredictMovieReviews.Models;
using System;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            // Load configuration
            IConfiguration config = new ConfigurationBuilder()
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .Build();

            // Load the trained model from the sentiment_model.zip file
            var context = new MLContext();
            var model = context.Model.Load(config["ModelSavedPath"], out var schema);

            // Use the model for predictions
            PredictSentiment(context, model);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }

    static void PredictSentiment(MLContext context, ITransformer model)
    {
        try
        {
            // Create a prediction engine
            var engine = context.Model.CreatePredictionEngine<MovieReview, SentimentPrediction>(model);

            // Sample movie reviews for prediction
            var reviews = new[]
            {
                new MovieReview { Text = "This movie was great!" },
                new MovieReview { Text = "I hated every moment of it." }
            };

            // Make predictions
            foreach (var review in reviews)
            {
                var prediction = engine.Predict(review);
                Console.WriteLine($"Review: \"{review.Text}\"");
                Console.WriteLine($"Predicted Sentiment: {prediction.Prediction}");
                Console.WriteLine();
            }

        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred during prediction: {ex.Message}");
        }
    }


}
