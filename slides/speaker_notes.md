# Speaker notes (target 15:00–16:00)

Planned total: 15:25. Questions come afterwards and do not count. The rubric stops talks at 16:00 and gives no timing points under 14:00, so rehearse with a timer at least twice.

* If a slide runs short, use the spare seconds on its figure: name the axes, then point at the number that matters.
* The rubric requires both of you to present and answer questions. A natural split: one of you takes slides 1–8, the other slides 9–16 with the live demo.
* Demo backup: if the notebook fails, open `notebooks/demo.html` (the same notebook, already run).

## 1. Delphi-lite  (0:30, starts at 0:00)

Hi, we are Ragnar Björn and Ólafur Sær. Our project applies Delphi, a transformer that was built to read people's health histories, to people's movie-rating histories. The question is in the subtitle: which movies will a user rate on their next active day, and does their own history help?

## 2. From health histories to movie histories  (1:00, starts at 0:30)

Delphi, published in Nature last year, is a GPT-2 that reads a person's diagnoses together with the age at which they happened, and predicts what comes next and when. A movie-rating history has the same shape: events with dates. So we ask a focused question: given everything a user rated up to the end of an active day, which movies will they rate on their next active day? And the part that makes it interesting: does the user's own history help beyond simply knowing what is popular at that time? That second part decides which baselines we need.

## 3. MovieLens-25M: ratings, not viewings  (0:40, starts at 1:30)

The data is MovieLens 25M from GroupLens: 25 million ratings by 162 thousand users, 1995 to 2019. Every rating is one event, whatever its stars. Two things to keep in mind. First, the timestamp is when the rating was entered, not when the movie was watched. Second, the licence allows research use with a citation but forbids redistribution, so our hand-in has a download script and no data.

## 4. Histories are short, bursty and batched  (0:55, starts at 2:10)

Looking at the histories changed our task definition. On the left: 56% of users rate everything on a single day, so there is no earlier day to learn from; we keep users with at least two active days. 57% of all ratings are typed in on the user's first day, the sign-up batch. On the right, ratings come in bursts: the busiest day, 2000-11-20, has 78,444 ratings. Same-day ratings have no meaningful order, so we treat them as ties.

## 5. The prediction task  (1:10, starts at 3:05)

Here is the exact task. For every held-out user we pick one target day at random among their active days after the first. The prediction time is the end of the last active day before it. The model sees only ratings from earlier days, at most the latest 256. The target is the set of movies rated on the target day; the median is 7 movies. Candidates are all 10,000 movies except those the user has already rated, because MovieLens users never rate a movie twice. We measure recall at 20: of the top 20 suggestions, how many were rated that day, divided by 20 or by the size of the target set if it is smaller.

## 6. Keeping the evaluation honest  (0:40, starts at 4:15)

Four rules keep this honest. We split users before anything else. The movie vocabulary and every baseline are fitted on training users only. Every decision, including the baseline's window, was made on validation users, and the test users were scored once at the end. And the model never sees the target day itself.

## 7. Delphi, adapted to movie days  (1:10, starts at 4:55)

The model is Delphi: GPT-2 where the position embedding is replaced by a sinusoidal encoding of the event time. Each movie has a log-rate; the next event is a competing-risks draw, so the loss is cross-entropy for which movie plus an exponential likelihood for when. We adapted four things. Time is the calendar day. Ties: Delphi's own tie masking hides same-day events from attention, but the current token still reaches its prediction through the residual connection; a unit test shows this leak. So we read every prediction from the last position on an earlier day. Tokens zero and one are reserved and never predicted, and we do not use Delphi's generation, whose stop token is death. Finally, we tried giving every movie a popularity baseline rate; it helped early in training but not at the end, so the final model does not use it.

## 8. Three baselines to beat  (0:40, starts at 6:05)

We compare against three baselines, all fitted on training users. POP ranks by all-time popularity. RECENT-POP ranks by what training users rated in the last 30 days up to the prediction day; we chose that window on validation among 12 variants, and it is the strongest baseline. ITEM-KNN is personalised: it suggests movies that co-occur with the user's history, but it ignores time.

## 9. One change at a time, decided on validation  (0:55, starts at 6:45)

This is the path from the baselines to the final model, all on validation users, one change per row. M1, a small Delphi with four layers and 128 dimensions, only matches RECENT-POP. Changing the time unit, the objective or the loss weighting did nothing, and a model three and a half times larger did not help either. What helped was training three times longer: that is M5, our final model. The popularity offset, M7 and M8, sped up early training, but its learned weight shrank from 0.49 to 0.31 and it ended below M5, so we dropped it.

## 10. Headline: Delphi beats recent popularity  (1:00, starts at 7:40)

On the 7,129 test users, scored once at the end, Delphi reaches recall at 20 of 0.158, against 0.149 for RECENT-POP. The paired difference on the same users is +0.009 plus or minus 0.005, about 6% relative. The three training seeds land between 0.156 and 0.158, and each one beats the baseline on its own. Recall at 10, recall at 50 and NDCG agree. All-time popularity is far behind at 0.117, and the time-blind ITEM-KNN at 0.136.

## 11. The gain needs the user's own history  (1:00, starts at 8:40)

Where does the gain come from? Left: we give the model only the latest k ratings. With a single rating it already scores 0.152, level with RECENT-POP: from the date and one film it has learned what is popular then. More of the user's history adds the rest, up to 0.159 at 32 ratings, and then it is flat. Right, two controls. Shuffling the order of the user's own ratings costs only 0.003, so which films matter more than their order. Giving the model another user's history, from just before the same date, drops it to 0.141, below the popularity baseline: the model personalises to whatever history it is given.

## 12. Where it fails  (1:10, starts at 9:40)

Where does it fail? Short histories: with fifteen ratings or fewer, RECENT-POP is better; Delphi's lead grows with the history. Gaps: next-day returns are a tie, and the biggest gain is when users come back after one to twelve months. Calendar: in 2015 to 2019, 31% of the test cases, Delphi is slightly behind. And the long tail: only 16% of the movies users rate next are among the 100 most popular, and like every method Delphi almost never finds the rest. In 42% of cases neither Delphi nor RECENT-POP finds a single movie; those are mostly small days of less popular films.

## 13. Live demo: held-out users  (2:40, starts at 10:50)

Now the demo, in the notebook. One: pick a held-out test user and step through their timeline, one day at a time, comparing Delphi with RECENT-POP. Two: open one step and look at what the user rated next, with hits highlighted. Three: a failure, and why. Four: type in a few movies we like and see what Delphi suggests. If the live demo fails, the same notebook is exported as HTML.

## 14. Extras: seeds, data size and cost  (0:40, starts at 13:30)

Some extras. Three seeds give 0.156 to 0.158, so the gain over the baseline is not seed noise. Data size: with the small ten-thousand-iteration budget, training on 10, 30 and 100 percent of the training users gives 0.144, 0.147 and 0.147: at that budget more users barely help; training longer is what helped. The final model has 2.1 million parameters, trains in about 29 minutes on a laptop GPU and answers one user in about 3 milliseconds on a CPU.

## 15. Limitations and next steps  (0:45, starts at 14:10)

Limitations. Ratings are not viewings, so a good suggestion can count as a miss. A UTC day can cut an evening in two, which makes some next-day targets easy. The split is by user, so the model and the baselines learned from training users' ratings after a test user's prediction day. We checked this: if no method may suggest a film that appeared after the prediction day, Delphi's gain over RECENT-POP stays at +0.009. A time-based split would still be stricter. And we only predict which movies, not when. Next steps: a temporal split, evaluating Delphi's waiting-time head, and generating synthetic histories from training users only.

## 16. Summary  (0:30, starts at 14:55)

To sum up: we adapted Delphi to movie-rating histories with a strict whole-day evaluation. Popularity in the last month is a strong baseline; Delphi beats it by a small but repeatable margin on held-out users, the gain depends on the user's own history, and the long tail is still out of reach. Thank you, we are happy to take questions.
