# How We Swipe: The Dataset

The `swipelogs.zip` file comprises 1338 `.log` (swipe data) and associated `.json` (metadata) files.
The uncompressed data takes about 1 GB of disk space.

**Note:** Our typing study was available "in the wild", so a few logs are ill-formed and even contain non-English words.
We recommend to use the log parsers provided in https://github.com/luileito/swipetest/tree/master/experiments

## Swipe log file example

This is an excerpt of `g1dgig1st0he01bmgdjo6crgds.log` file (space-separated format):

```csv
sentence timestamp keyb_width keyb_height event x_pos y_pos x_radius y_radius angle word is_err
force_instrumental_reliance_replacement 1576480557856 360 205 touchstart 140 100 1.3324081897736 1.3324081897736 0 force 0
force_instrumental_reliance_replacement 1576480557864 360 205 touchmove 140 100 1.4989590644836 1.4989590644836 0 force 0
force_instrumental_reliance_replacement 1576480557879 360 205 touchmove 140 100 1.1658570766449 1.1658570766449 0 force 0
force_instrumental_reliance_replacement 1576480557898 360 205 touchmove 142 99 1.1658570766449 1.1658570766449 0 force 0
force_instrumental_reliance_replacement 1576480557920 360 205 touchmove 167 90 1.3324081897736 1.3324081897736 0 force 0
force_instrumental_reliance_replacement 1576480557945 360 205 touchmove 232 67 1.4989590644836 1.4989590644836 0 force 0
force_instrumental_reliance_replacement 1576480557963 360 205 touchmove 253 60 1.1658570766449 1.1658570766449 0 force 0
force_instrumental_reliance_replacement 1576480557982 360 205 touchmove 266 56 0.99930608272552 0.99930608272552 0 force 0
force_instrumental_reliance_replacement 1576480557991 360 205 touchmove 270 54 1.3324081897736 1.3324081897736 0 force 0
force_instrumental_reliance_replacement 1576480558007 360 205 touchmove 286 48 1.4989590644836 1.4989590644836 0 force 0
```

The columns comprise the following information:

* `sentence` (string) Prompted sentence. Words are separated by underscores.
* `timestamp` (integer) Time (in milliseconds) at which the event was created.
* `keyb_width` (integer) Width of the rendered virtual keyboard.
* `keyb_height` (integer) Height of the rendered virtual keyboard.
* `event` (string) Name of the event.
* `x_pos` (integer) X coordinate of the touch point relative to the keyboard width.
* `y_pos` (integer) Y coordinate of the touch point relative to the keyboard height.
* `x_radius` (float) X radius of the ellipse that most closely circumscribes the touch contact area.
* `y_radius` (float) Y radius of the ellipse that most closely circumscribes the touch contact area.
* `angle` (float) Rotation angle, in degrees, of the contact area ellipse defined by `x_radius` and `x_radius`.
* `word` (string) Swiped word.
* `is_err` (bool) Flag indicating whether the swiped word was entered correctly (0) or not (1).

More info about touch events: https://developer.mozilla.org/en-US/docs/Web/API/Touch_events

## User metadata file example

This is the content of `g1dgig1st0he01bmgdjo6crgds.json` file:
```json
{"gender":"Male","age":40,"nationality":"ES","familiarity":"Everyday","englishLevel":"Advanced","dominantHand":"Right","swipeHand":"Right","swipeFinger":"Index","screenWidth":360,"screenHeight":720,"devicePixelRatio":3,"maxTouchPoints":5,"platform":"Linux armv8l","vendor":"Google Inc.","referal":"","language":"en-US,en;q=0.9","userAgent":"Mozilla\/5.0 (Linux; Android 7.1.1; K10) AppleWebKit\/537.36 (KHTML, like Gecko) Chrome\/79.0.3945.79 Mobile Safari\/537.36","timestamp":1576480549}
```

## Processed files

We provide several processed files to ease further analyses.
These processed files refer to the subset of 909 users that we reported in our paper.

The procedure we performed is the following:
- Select users who submitted 5 or more sentences (one third of the requested sentences).
- Select mobile users (`maxTouchPoints > 2`, screen size criteria: `width < 600` and `500 < height < 900`).
- Remove users having `NA`s in their demographic information.

### Metadata

Log metadata, such as user demographics, language, device platform, etc. are provided in `metadata.tsv` file (tab-separated format).

```csv
age	dominant_hand	english_level	familiarity	gender	language	max_touch_points	nationality	screen_height	screen_width	swipe_finger	swipe_hand	timestamp	uid	vendor
19	right	beginner	sometimes	female	en	5	ae	640	360	thumb	right	1583309231	0q1sat7ola2osk0ohdtciaboba	google inc.
19	left	native	rarely	female	en	5	us	869	412	index	right	1583136907	124p7a10vadpv7d0rk37oiv3g5	google inc.
44	left	beginner	rarely	female	en	5	mx	640	360	index	both	1582648583	19hubm5jt63707ejbcnkimlr08	google inc.
18	right	intermediate	never	female	mt	5	mt	892	412	thumb	right	1582555916	20sa5uaiit0t55or9miatm70tv	google inc.
38	right	advanced	rarely	male	es	5	mx	846	412	thumb	both	1582073116	2r7mdfhcjesci8trr5afg2akq5	google inc.
18	right	advanced	sometimes	male	en	5	ca	667	375	thumb	right	1581726141	2sd22gsn7kdnhs3mrmh93j7hvk	apple computer, inc.
25	right	native	rarely	female	en	5	us	896	414	other	right	1582827386	3a064eetjpjpr55674f3qc5t55	apple computer, inc.
19	left	native	rarely	male	en	5	us	869	412	thumb	right	1582557536	3fmgdlooume5t7fu6dula4rn8m	google inc.
22	right	native	often	female	en	5	us	896	414	thumb	right	1583387995	4ikrjsdgdlgfhvsbekfpo8605g	apple computer, inc.
```

The file has 15 columns:

* `uid` (string) User ID. It unequivocally links all dataset files, e.g. swipe logs, metadata, etc.
* `timestamp` (integer) Time (in seconds) at which the user entered the study.
* `nationality` (string) Self-reported user nationality, according to the ISO 3166-1 alpha-2 standard: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
* `age` (integer) Self-reported user age. Possible values are between 18 and 99.
* `gender` (string) Self-reported user gender. Possible values are: "male", "female", "other".
* `dominant_hand` (string) Self-reported dominant hand. Possible values are: "left", "right".
* `swipe_hand` (string) Self-reported input hand. Possible values are: "left", "right", "both".
* `swipe_finger` (string) Self-reported input finger. Possible values are: "index", "thumb", "other".
* `english_level` (string) Self-reported user gender. Possible values are: "beginner", "intermediate", "advanced", "native".
* `familiarity` (string) Self-reported familiarity with shape-writing input. Possible values are: "never", "rarely", "sometimes", "often", "everyday".
* `language` (string) Preferred user language, according to the HTTP Accept-Language header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Language
* `max_touch_points` (integer) Maximum number of simultaneous touch contact points supported by the device.
* `screen_height` (integer) Height of the device screen, in CSS pixels.
* `screen_width` (integer) Width of the device screen, in CSS pixels.
* `vendor` (string) Navigator vendor property. Possible values are "google inc." or "apple computer, inc.".

### Sentence-level performance stats

This is an excerpt of `stats-sentences.tsv` file (tab-separated format):

```csv
good_wpm_swipe	sentence	fail_wpm_classic	fail_wpm_swipe	fail_interval_time	fail_length	fail_time	good_length	fail_dtw	good_interval_time	good_dtw	good_time	good_wpm_classic	wer	username	dataset
18.184573420215184	they_have_capacity_now	15.933609958506224	9.95850622406639	nan	1409.5558752582206	3012.5	904.556376919928	10498.246089818047	1703.5	3660.1316217387975	1648.0	20.0030307622367	0.25	0q1sat7ola2osk0ohdtciaboba	enron
39.267015706806276	what_a_jerk				nan	nan	453.5682207227957	nan	nan	2146.824630286267	1528.0	35.3403141361256560.0	0q1sat7ola2osk0ohdtciaboba	enron
23.170496234794363	what_is_the_mood				nan	nan	284.2313386802028	nan	1458.0	1512.468340490222	890.5	18.53639698783549	0.0	0q1sat7ola2osk0ohdtciaboba	enron
15.671251958906494	im_on_a_plane	19.524894240156197	19.5248942401562		nan	nan	103.51568765980456	nan	3994.5	452.42589317380464442.0	11.49225143653143	0.25	0q1sat7ola2osk0ohdtciaboba	enron
12.061513719971856	mcdonald_click_adventures_amenities				nan	nan	396.0785894109886	nan	1758.0	5906.483283114141	2984.5	21.10764900995075	0.0	0q1sat7ola2osk0ohdtciaboba	rand
22.253129346314328	we_are_all_fragile				nan	nan	218.1191927999088	nan	1478.0	1485.3408015189236	669.0	20.027816411682892	0.0	0q1sat7ola2osk0ohdtciaboba	enron
8.916960802526471	mathematics_komatsu_mold_shop				nan	nan	712.6508009226677	nan	2663.0	4821.69478236841	3218.5	12.929593163663384	0.0	0q1sat7ola2osk0ohdtciaboba	rand
14.232624337787616	arms_sichuan_asks_blogs				nan	nan	589.5511226552718	nan	nan	5399.855165735805	3772.0	17.07914920534514	0.0	0q1sat7ola2osk0ohdtciaboba	rand
4.908644668666485	generates_kentucky_holidays_etrade				nan	nan	947.2273284276482	nan	5530.5	13442.322231641021	10055.0	8.835560403599674	0.0	0q1sat7ola2osk0ohdtciaboba	rand
```

The file has 16 columns:

* `username` (string) User ID.
* `sentence` (string) Prompted sentence. Words are separated by underscores.
* `dataset` (string) Dataset which the sentence belongs to. Possible values are: "enron" (memorable sentences), "rand" (randomly generated sentences).
* `good_length` (float) Median swipe length for the _successfully entered_ words.
* `good_time` (float) Median swipe time for the _successfully entered_ words.
* `good_interval_time` (float) Median interval time for the _successfully entered_ words.
* `good_dtw` (float) Median swipe error of _successfully entered_ words, in pixels. It is computed as the elastic matching score against the ideal swipe trajectory, according to the Dynamic Time Warping (DTW) algorithm.
* `good_wpm_classic` (float) Words per minute according to http://www.yorku.ca/mack/RN-TextEntrySpeed.html for the _successfully entered_ words. This is the WPM we report in our paper.
* `good_wpm_swipe` (float) Words per minute computed by dividing the number of entered words by the overall time (including both swipe times and interval times) for the _successfully entered_ words.
* `wer` (float) Median "a priori" word error rate, computed as the number of (unique) failed words divided by the number of reference words. This value is not reliable because users had to enter all words successfully in order to advance to the next sentence.
* `fail_length` (float) Median swipe length for the _failed_ words.
* `fail_time` (float) Median swipe time for the _failed_ words.
* `fail_interval_time` (float) Median interval time for the _failed_ words.
* `fail_dtw` (float) Median swipe error of _failed_ words, in pixels. It is computed as the elastic matching score against the ideal swipe trajectory, according to the Dynamic Time Warping (DTW) algorithm.
* `fail_wpm_classic` (float) Words per minute, according to http://www.yorku.ca/mack/RN-TextEntrySpeed.html for the _failed_ words.
* `fail_wpm_swipe` (float) Words per minute computed by dividing the number of entered words by the overall time (including both swipe times and interval times) for the _failed_ words.

### Word-level performance stats

This is an excerpt of `stats-words.tsv` file (tab-separated format):

```csv
dataset	dtw	is_failed	length	sentence	time	username	word
enron	4853.106069030182	0	1078.9805063431345	they_have_capacity_now	1709	0q1sat7ola2osk0ohdtciaboba	they
enron	2467.1571744474127	0	730.1322474967213	they_have_capacity_now	1587	0q1sat7ola2osk0ohdtciaboba	have
enron	17806.80973715143	0	1685.6432266743946	they_have_capacity_now	5574	0q1sat7ola2osk0ohdtciaboba	capacity
enron	2142.9952767392892	0	333.4053003465969	they_have_capacity_now	921	0q1sat7ola2osk0ohdtciaboba	now
enron	18134.877696786567	1	2381.1457347329333	they_have_capacity_now	5001	0q1sat7ola2osk0ohdtciaboba	capacity
enron	2861.6144828495258	1	437.9660157835077	they_have_capacity_now	1024	0q1sat7ola2osk0ohdtciaboba	capacity
enron	2116.483479831384	0	506.3568403260237	what_a_jerk	1399	0q1sat7ola2osk0ohdtciaboba	what
enron	2177.1657807411507	0	400.77960111956764	what_a_jerk	1657	0q1sat7ola2osk0ohdtciaboba	jerk
enron	1854.654769488807	0	495.8353143406047	what_is_the_mood	1176	0q1sat7ola2osk0ohdtciaboba	what
```

The file has 8 columns:

* `username` (string) User ID.
* `dataset` (string) Dataset which the word belongs to. Possible values are: "rand2k" (highly frequent words), "rand3k" (common words), "rand5k" (infrequent words), "rand0" (out-of-vocabulary words).
* `word` (string) Swiped word.
* `is_failed` (bool) Whether the word was swiped successfully (0) or not (1).
* `length` (float) Swipe length, in pixels.
* `time` (integer) Swipe time, in milliseconds.
* `dtw` (float) Swipe error, in pixels. It is computed as the elastic matching score against the ideal swipe trajectory, according to the Dynamic Time Warping (DTW) algorithm.

## More resources

See https://github.com/luileito/swipetest for the source code of our online study, the phrase sets, and the processing scripts we used in our experiments.

# Citation

Please cite us using the following reference:

- L. A. Leiva, S. Kim, W. Cui, X. Bi, A. Oulasvirta.
  **How We Swipe: A Large-scale Shape-writing Dataset and Empirical Findings.**
  *Proc. MobileHCI, 2021.*

```bib
@InProceedigs{swipe_dataset,
  author    = {Luis A. Leiva and Sunjun Kim Wenzhe Cui and Xiaojun Bi and Antti Oulasvirta},
  title     = {How We Swipe: A Large-scale Shape-writing Dataset and Empirical Findings},
  booktitle = {Proc. MobileHCI},
  year      = {2021},
}
```
