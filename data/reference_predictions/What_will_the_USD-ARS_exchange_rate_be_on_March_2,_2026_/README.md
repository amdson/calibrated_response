This README file describes how to interpret the data in this zip file.
If you have any questions or comments, please contact the Metaculus team: support@metaculus.com.

Metadata:
This data was exported on 2026-02-13 22:22:25.274728+00:00
Contains the data for 1 questions
Contains the data for 341 aggregate forecasts


# File: README.md

This File


# File: `question_data.csv`

This file contains summary data of the questions specific to this dataset

### columns:

**`Question ID`** - the id of the question. This is not the value in the URL.
**`Question URL`** - the URL of the question.
**`Question Title`** - the title of the question.
**`Post ID`** - the id of the Post that this question is part of. This is the value in the URL.
**`Post Curation Status`** - the curation status of the Post.
**`Post Published Time`** - the time the Post was published.
**`Default Project`** - the name of the default project (usually a tournament or community) for the Post.
**`Default Project ID`** - the id of the default project for the Post.
**`Categories`** - a list of category names that this question belongs to.
**`Leaderboard Tags`** - a list of leaderboard tag names associated with this question.
**`Label`** - for a group question, this is the sub-question object.
**`Question Type`** - the type of the question. Binary, Multiple Choice, Numeric, Discrete, or Date.
**`MC Options (Current)`** - the current options for a multiple choice question, if applicable.
**`MC Options (All)`** - the options for a multiple choice question across all time, if applicable.
**`MC Options History`** - the history of options over time. Each entry is a isoformat time and a record of what the options were at that time.
**`Lower Bound`** - the lower bound of the forecasting range for a continuous question.
**`Open Lower Bound`** - whether the lower bound is open.
**`Upper Bound`** - the upper bound of the forecasting range for a continuous question.
**`Open Upper Bound`** - whether the upper bound is open.
**`Continuous Range`** - the locations where the CDF is evaluated for a continuous question.
**`Open Time`** - the time when the question was opened for forecasting.
**`CP Reveal Time`** - the time when the community prediction is revealed.
**`Scheduled Close Time`** - the time when forecasting ends.
**`Actual Close Time`** - the earlier of the scheduled close time and the time when the resolution became known.
**`Resolution`** - the resolution for the question.
**`Resolution Known Time`** - the time when the resolution became known.
**`Include Bots in Aggregates`** - whether bots are included in the aggregations by default.
**`Question Weight`** - the weight of the question in the leaderboard.


# File: `forecast_data.csv`

This file contains the user and aggregation forecast data for the questions in this dataset.

### columns:

**`Question ID`** - the id of the question this forecast is for. Cross-reference with 'Question ID' in `question_data.csv`.
**`Forecaster ID`** - the id of the forecaster.
**`Forecaster Username`** - the username of the forecaster or the aggregation method.
**`Is Bot`** - if user is bot.
**`Start Time`** - the time when the forecast was made.
**`End Time`** - the time when the forecast ends. If not populated, the forecast is still active. Note that this can be set in the future indicating an expiring forecast.
**`Forecaster Count`** - if this is an aggregate forecast, how many forecasts contribute to it.
**`Probability Yes`** - the probability of the binary question resolving to 'Yes'
**`Probability Yes Per Category`** - a list of probabilities corresponding to each option for a multiple choice question. Cross-reference 'MC Options (All)' in `question_data.csv`. Note that a Multiple Choice forecast will have None in places where the corresponding option wasn't available for forecast at the time. Note: geometric_means always display values here in pmf form.
**`Continuous CDF`** - the value of the CDF (cumulative distribution function) at each of the locations in the continuous range for a continuous question. Cross-reference 'Continuous Range' in `question_data.csv`.
**`Probability Below Lower Bound`** - the probability of the question resolving below the lower bound for a continuous question.
**`Probability Above Upper Bound`** - the probability of the question resolving above the upper bound for a continuous question.
**`5th Percentile`** - the 5th percentile of forecast for a continuous question.
**`25th Percentile`** - the 25th percentile of forecast for a continuous question.
**`Median`** - the median of forecast for a continuous question.
**`75th Percentile`** - the 75th percentile of forecast for a continuous question.
**`95th Percentile`** - the 95th percentile of forecast for a continuous question.
**`Probability of Resolution`** - the actual probability assigned to the Resolution of the question, if resolved. This is the value used in scoring. Cross reference 'Resolution' in `question_data.csv`.
**`PDF at Resolution`** - the height of the PDF (probability density function) value at the resolution for a continuous question. This is the value that will show on the continuous range in the prediction interface.
