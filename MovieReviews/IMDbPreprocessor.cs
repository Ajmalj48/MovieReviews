using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovieReviews
{
    public static class IMDbPreprocessor
    {
        public static void PreprocessAndConvertToCSV(MLContext context, IConfiguration config)
        {
            string datasetDirectory = config["IMDbDatasetDirectory"]; // Path to the directory containing the extracted IMDb dataset
            string csvFilePath = config["PreprocessedCSVPath"] + "\\imdb_reviews.csv"; // Path to save the preprocessed CSV file

            StringBuilder csvContent = new StringBuilder();
            csvContent.AppendLine("Text,Sentiment");

            // Process positive reviews
            string positiveReviewsPath = Path.Combine(datasetDirectory, "aclImdb", "train", "pos");
            ProcessReviews(positiveReviewsPath, csvContent, "positive");

            // Process negative reviews
            string negativeReviewsPath = Path.Combine(datasetDirectory, "aclImdb", "train", "neg");
            ProcessReviews(negativeReviewsPath, csvContent, "negative");

            // Save CSV file
            File.WriteAllText(csvFilePath, csvContent.ToString());

            Console.WriteLine("Preprocessing and conversion to CSV completed successfully.");
        }

        static void ProcessReviews(string directoryPath, StringBuilder csvContent, string sentiment)
        {
            foreach (string file in Directory.EnumerateFiles(directoryPath, "*.txt"))
            {
                string text = File.ReadAllText(file);
                // Clean text if needed (e.g., remove special characters, HTML tags)
                text = CleanText(text);
                // Append to CSV
                csvContent.AppendLine($"\"{text}\",{sentiment}");
            }
        }

        static string CleanText(string text)
        {
            // Implement text cleaning logic as needed
            // For example, remove special characters, HTML tags, etc.
            // This depends on the specific requirements of your ML model
            return text;
        }
    }
}
