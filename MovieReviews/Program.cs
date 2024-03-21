using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using MovieReviews;
using MovieReviews.Models;

class Program
{
    static void Main(string[] args)
    {
        // Load configuration
        IConfiguration config = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .Build();

        var context = new MLContext();

        // Preprocess and convert IMDb dataset to CSV
        //IMDbPreprocessor.PreprocessAndConvertToCSV(context,config);

        // Load data
        var data = context.Data.LoadFromTextFile<MovieReview>(config["PreprocessedCSVPath"] + "\\imdb_reviews.csv", separatorChar: ',', hasHeader: true);

        // Data preprocessing (e.g., tokenization, normalization)
        var preProcessedData = PreprocessData(context, data);

        // Split data into train and test sets
        var trainTestSplit = context.Data.TrainTestSplit(preProcessedData, testFraction: 0.2);
        var trainData = trainTestSplit.TrainSet;
        var testData = trainTestSplit.TestSet;

        // Train the model
        var model = TrainModel(context, trainData);

        // Evaluate the model
        EvaluateModel(context, model, testData);

        // Save the model
        SaveModel(context, model, config["ModelSavePath"] + "\\sentiment_model.zip");
    }

    static IDataView PreprocessData(MLContext context, IDataView data)
    {
        var pipeline = context.Transforms.Conversion.ConvertType("Sentiment", "Sentiment", DataKind.Boolean)
            .Append(context.Transforms.Text.FeaturizeText("Features", nameof(MovieReview.Text)))
            .Append(context.Transforms.NormalizeMinMax("Features"));

        return pipeline.Fit(data).Transform(data);
    }

    static ITransformer TrainModel(MLContext context, IDataView data)
    {
        var pipeline = context.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(MovieReview.Sentiment))
            .Append(context.Transforms.NormalizeMinMax(inputColumnName: "Features", outputColumnName: "Features"))
            .Append(context.BinaryClassification.Trainers.LbfgsLogisticRegression());

        return pipeline.Fit(data);
    }

    static void EvaluateModel(MLContext context, ITransformer model, IDataView testData)
    {
        var predictions = model.Transform(testData);
        var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

        Console.WriteLine($"Accuracy: {metrics.Accuracy}");
        Console.WriteLine($"F1 Score: {metrics.F1Score}");
    }

    static void SaveModel(MLContext context, ITransformer model, string filePath)
    {
        context.Model.Save(model, null, filePath);
        Console.WriteLine($"Model saved to: {filePath}");
    }
}
