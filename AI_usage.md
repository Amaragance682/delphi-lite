# AI usage

Claude Code (Anthropic, model Claude Opus 5.5) helped write the code, ran parallel experiments and drafted the report as well as acting like a tutor on this project and connecting topics to previous assignments in this course.

- MovieLens timestaps record when a rating was entered, not when a film was actually watched, this lead to prediciting the next rating would mostly mean predicting the order of a batch typed in one sitting, claude suggested treating whole UTC days as the time unit and rating on the same day treated as unordered ties, this was only done for users with at least two active days with each validation and test user getting one randomly selected target day.

- Same-day leak in Delphi's own tie handling, due to Delphi's `mask_ties` option hiding same-day events from attention, the current token still reaches its own prediction through a residual connection, so a rating can help predict another rating made on the same day, claude suggested fixing this by making every prediction for day *d* being read from the last position on a strictly earlier day, in both training and evaluation, with appropriate unit tests for it.

- The baseline we started with (POP) was way too weak, at 0.117 val and recall at 20%, beating it doen't say much, claude suggested RECENT-POP which gave 0.143 on val which our own first model only matched it.

- An idea that only helped early in training, by adding log-popularity prior to M1's scores which raised validation recall@20 from 0.145 to 0.153 which suggested building it into the model, we fixed this with a learned popularity offset on every log-rate. It sped up early training but at convergence it didn't really help at all.

- Gain could come from learning calendar popularity rather than a user's tase, we fixed this with checks on the test users, which involved shuffling the order of a user's own rating or  giving the model another user's history, claude explained to us why a score alone being what it is isn't always attributed to what we think, instead the gain may come from somewhere else.

- A lot of our code was messy and hard to reproduce across different operating systems, so claude refactored a lot of our code for compatability and reusability, there were also silent errors and incorrect results that claude explained and in some cases fixed for us.

- Finalizing the report and making it fit in only 2 pages was largely done by claude, being able to fit everything from 5 pages originally into 2 pages was difficult and claude was used to discuss what could be more concise, cut off and trade-off's from different sections.

- A sample presentation has been made by claude for comparison for our own presentation we have yet to make (13/11/2026), we will likely use a few talking points from it and its presentation sequence of sections.

- A disk failure mid write to git caused repository corruption, claude helped us recover from it.

- Claims against the report and data were checked by claude to ensure that our report was accurate and up to date.