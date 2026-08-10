# Qualitative comparison

13 test examples selected by behavior (not by score) to illustrate distinct phenomena. Diagnostics shown per output: `dup3` = duplicate-trigram rate, `novel2` = novel-bigram rate vs. the article, `unsup` = content words absent from the article, `oov` = tokens outside the LSTM's 50k vocabulary, `R1` = ROUGE-1 F1.


> **Categories below were assigned by reading each output against its full source article**, not
> inferred from the diagnostics. Every claim of fabrication was checked by searching the source
> text: three were confirmed (§3, §9, §11) and one suspected case was **refuted** (§8).
>
> Two patterns hold across all thirteen. `dup3` is **0.00 in every LSTM output**, including the one
> selected as *most repetitive*: repeated-trigram blocking removes literal repetition
> entirely, so what survives is incoherence rather than repetition (§6). And `oov` is **0.00 in every LSTM
> output**, including the one selected as *highest OOV rate*, because `<unk>` is suppressed at
> generation; the rare-word failure relocates to substitution (§3) and omission (§7, §13).


---

## 1. LSTM's strongest case

**Example id:** `7fd2aa8042a8c13a7fe70e136d7611f08158d388`, source length 592 tokens

**Source (truncated for display):**

> police are investigating links between the hatton garden gems heist and a strikingly similar raid on a berlin
> bank two years ago. german detectives have been ‘closely monitoring developments’ in the £60 million london
> case and are expected to make contact with scotland yard this week. in particular, they are keen to find out
> whether any dna was recovered from the scene of the easter weekend raid and, crucially, whether it matches
> forensic samples found in a tunnel used in a £8.3 million robbery at the volksbank in steglitz, berlin, in
> january 2013. the culprits behind the hatton garden raid (pictured) may have carried out a similarly audacious
> burglary just two years earlier in berlin, it has emerged . in a case which has strikingly similar traits to
> the £60m heist in london, thieves targeted 294 security vaults during a break-in at the volksbank in steglitz,
> berlin, in january 2013 . th [...]

**Reference:**

> police investigating links between hatton garden heist and berlin bank raid . german detectives expected to
> make contact with scotland yard this week . they are keen to find out whether any dna was recovered from the
> scene .

**lstm_beam:**
`R1=72.1` `dup3=0.00` `novel2=0.15` `unsup=0.00` `oov=0.00` `len=56`

> police are investigating links between the hatton garden. they are keen to find out whether any dna was
> recovered from the scene of the easter weekend raid. they have been' closely monitoring developments' in the £
> 60 million london case and are expected to make contact with scotland yard this week.

**llm_B_zeroshot:**
`R1=35.1` `dup3=0.01` `novel2=0.11` `unsup=0.00` `oov=0.04` `len=84`

> German detectives are closely monitoring developments in the £60 million London case and are expected to make
> contact with Scotland Yard this week. The culprits behind the Hatton Garden raid may have carried out a
> similarly audacious burglary in Berlin two years earlier. Thieves targeted 294 security vaults during a break-
> in at the Volksbank in Steglitz, Berlin, in January 2013. The London robbers stole jewels and cash from 72
> boxes at the Hatton Garden safe deposit centre.

**Error category (LSTM):** Extractive echo, faithful (unsup 0.00). Two reference sentences reproduced almost verbatim, which is why it scores 72.1. One defect: the opening clause stops at *"links between the hatton garden"*, dropping *heist and a berlin bank raid*, the object of the comparison is lost. Grammatical, not factual.

**Error category (LLM):** Content-selection mismatch, no factual error (unsup 0.00). Every claim is in the article; it leads on the German detectives and adds the Volksbank vault count and the 72 boxes, none of which the editor kept. 84 tokens against a ~40-token reference.


---

## 2. LSTM's weakest case

**Example id:** `c965689f5be9c8fa5c1aab5c28c014344f4cab0f`, source length 970 tokens

**Source (truncated for display):**

> i am not vain, rich or brave enough to face plastic surgery, when my mother rang for her weekly chat and i was
> feeling ancient after spotting a new wrinkle, i probed her for beauty advice. if i was feeling old, she, with
> a nearly 40-year-old daughter, must be feeling ancient - yet she looked great. at 65, she could easily pass
> for ten years younger. scroll down for video . sarah's face was transformed after the exercises. she's
> pictured before (left) and after (right) 'i thank my genes, the fact i've never tried a cigarette and the
> facial stretches i've been doing for 30 years,' she said. ah, those. i remember my mum slathered in vaseline,
> gurning in a club sport tracksuit while i ate my coco pops, invariably wearing the deely bobbers i insisted on
> sleeping in between 1983 and 85. so they worked? my mother swore they did. facial exercises have been around
> as long as youth has been consid [...]

**Reference:**

> sarah ivens, 39, tries the 'furrow smoother', 'lip lift' and 'nose transformer' . exercises prescribed by
> carole maggio, the la-based creator of facercise . mother-of-two noticed her lips were fuller after two weeks
> of exercises . experienced muscle burn and soreness during and after the moves .

**lstm_beam:**
`R1=2.6` `dup3=0.00` `novel2=0.23` `unsup=0.12` `oov=0.00` `len=36`

> " i am not vain, rich or brave enough to face plastic surgery, " she says. she says she had to hold this
> tension for 30 seconds while staring at the ceiling.

**llm_B_zeroshot:**
`R1=32.1` `dup3=0.00` `novel2=0.18` `unsup=0.02` `oov=0.09` `len=95`

> Facial exercises have been around since 1710, with the first official guide published in Paris outlining the
> beauty routine of Ninon de l'Enclose, a courtesan famed for her lasting beauty. Carole Maggio, the LA-based
> creator of Facercise, is the go-to expert today with a celebrity and royalty-packed client list. The Furrow
> Smoother involves spreading fingers across the centre of the forehead and pulling down while pushing brows up
> for 30 seconds. The Lip Lift requires a Joker smile that moves up and down an inch using visualisation.

**Error category (LSTM):** Lead-bias miss + total entity omission. The output is the article's first-person opening aside, quoted verbatim, as if it were the summary. None of the reference's four points appears and no named entity survives, *sarah ivens*, *carole maggio* and *facercise* are all absent. R1 2.6 is the floor of the set.

**Error category (LLM):** Content-selection mismatch + over-length. Recovers two reference points (Maggio, Facercise, and both named exercises) but opens on the 1710 history the editor omitted. 95 tokens against ~50.


---

## 3. Largest LLM advantage

**Example id:** `b725b494ff80fb8fe21c69ca7f367562e95b516d`, source length 1287 tokens

**Source (truncated for display):**

> a runner up of brazil's miss bumbum competition has branded the contest's winner 'an attention-seeking sl*t'
> after she posted a photograph of her naked body painted with an image of the virgin mary for easter. claudia
> alende, 21, who came second in last year's beauty pageant, launched a scathing attack on current title-holder
> indianara carvalho, claiming the model had 'revealed who she truly is'. miss carvalho, 23, posted the racy
> picture to her instagram account on good friday - the day the country's 130million catholics solemnly
> remembered the crucifixion and death of christ. scroll down for video . racy: indianara carvalho posted the
> picture of her naked body painted with an image of the virgin mary on good friday . slammed: claudia alende,
> who came second in the competition, launched a scathing attack on miss carvalho . the image, one of a series
> of nude shots she shared over the eas [...]

**Reference:**

> indianara carvalho posted photo of body painted with image of virgin mary . runner-up claudia alende has
> branded winner an 'attention-seeking sl*t' . catholic alende refused photoshoot 'out of respect for god and my
> family' .

**lstm_beam:**
`R1=11.3` `dup3=0.00` `novel2=0.11` `unsup=0.05` `oov=0.00` `len=45`

> claudia james, 21, came second in last year's beauty pageant. she was accompanied with a prayer which
> including the words:' good friday. lord, on this day i ask for peace, love, wisdom and strength'.

**llm_B_zeroshot:**
`R1=53.6` `dup3=0.01` `novel2=0.20` `unsup=0.04` `oov=0.07` `len=92`

> Claudia Alende, 21, launched a scathing attack on Miss Bumbum winner Indianara Carvalho, calling her an
> 'attention-seeking sl*t'. Indianara Carvalho, 23, posted a picture of her naked body painted with an image of
> the Virgin Mary on Good Friday. The image was accompanied by a prayer asking for peace, love, wisdom, and
> strength. Claudia Alende claimed she was asked to do the photoshoot but refused 'out of respect for God and
> for my family'.

**Error category (LSTM):** **Entity fabrication, verified.** *"claudia james"* for *claudia alende*: the token *james* does not occur anywhere in the 1,088-word article. The main event, the Virgin Mary body paint, is dropped entirely. This is the substitution failure that the 0.000 OOV rate conceals.

**Error category (LLM):** Faithful and aligned. All three reference points covered, each traceable to the source. Mild over-length (92 tokens against ~45).


---

## 4. LSTM beats the LLM

**Example id:** `b8e4102f18ee377a019f96c14759797fa3c19819`, source length 836 tokens

**Source (truncated for display):**

> a pregnant model whose sexy selfies have turned her into an overnight internet sensation has snapped yet
> another another lingerie-clad photo of herself flaunting a barely-there baby bump - just 10 days before she is
> due to give birth. sarah stage, a 30-year-old underwear model and animal rights activist from los angeles, has
> documented her changing figure via her instagram page throughout her pregnancy, earning herself a huge number
> of fans - as well as a fair share of critics - in the process. although the mother-to-be came under fire last
> month as more and more critics spoke out against her unusually trim and toned figure, claiming it could be
> doing damage to her unborn child, their comments haven't stopped the model from sharing numerous photos of her
> tight abs with her 1.5 million instagram followers. scroll down for video . so soon: pregnant model sarah
> stage snapped this photo of h [...]

**Reference:**

> sarah stage, 30, has documented her changing figure via her instagram page throughout her pregnancy . as her
> pregnancy has progressed, the model has come under fire from critics who claim she is 'unhealthily' trim and
> toned .

**lstm_beam:**
`R1=57.1` `dup3=0.00` `novel2=0.00` `unsup=0.00` `oov=0.00` `len=28`

> sarah stage, a 30-year-old underwear model and animal rights activist from los angeles, has documented her
> changing figure via her instagram page throughout her pregnancy.

**llm_B_zeroshot:**
`R1=26.3` `dup3=0.00` `novel2=0.33` `unsup=0.05` `oov=0.02` `len=85`

> Sarah Stage, a 30-year-old underwear model, has shared a photo of her barely-there baby bump 10 days before
> her due date. The model, who has 1.5 million Instagram followers, posted the picture on Monday with the
> caption '10 days until we meet #babyjames.' The photo has been liked nearly 45,000 times and has earned nearly
> 5,000 comments. Sarah Stage is due to give birth to her son, Baby James, in 10 days.

**Error category (LSTM):** Extractive echo, faithful, and the clearest metric artefact in the set (novel-bigram 0.00). It reproduces the reference's first sentence nearly word for word and scores 57.1 while covering only half of it; the critics angle is omitted.

**Error category (LLM):** Content-selection mismatch, no error of fact. Every claim checks out; it reports the photo, the follower count and the due date rather than the criticism, and scores 26.3, **less than half the LSTM, for choosing different true facts**. This is §5's central example of what ROUGE rewards.


---

## 5. LSTM beats the LLM

**Example id:** `065cb87a025be1dd2d212a7190ba0100f80394fc`, source length 646 tokens

**Source (truncated for display):**

> patients are being put at risk because paramedics are leaving firemen to deal with 999 calls, healthcare
> leaders warn. many firemen have not been trained in basic first aid - yet they are being left to care for
> seriously ill elderly patients for hours while they wait for an ambulance. the fire brigades union say the
> practice is irresponsible and dangerous but is becoming increasingly common because the ambulance service is
> so overstretched. the ambulance service is becoming increasingly reliant on firemen to attend life-threatening
> emergencies . the fire brigade union say the practice is irresponsible and dangerous and becoming more common
> . senior union officials say fire engines are routinely being turned into makeshift ambulances to ferry
> patients to a&e. they normally carry a few blankets, oxygen masks to protect against smokes and a
> defibrillator - to restart the heart. in one case  [...]

**Reference:**

> the fire brigades union say practice is irresponsible and dangerous . but becoming more common as ambulance
> service is so overstretched . ambulance service facing huge demand from rising and aging population . led to
> them relying more on firemen to attend life threatening emergencies .

**lstm_beam:**
`R1=56.5` `dup3=0.00` `novel2=0.00` `unsup=0.00` `oov=0.00` `len=45`

> the fire brigades union say the practice is irresponsible and dangerous and becoming more common. the
> ambulance service is facing unprecedented demand from the rising and aging population and in some areas 999
> calls are up by 15 per cent on last year.

**llm_B_zeroshot:**
`R1=30.4` `dup3=0.00` `novel2=0.30` `unsup=0.03` `oov=0.00` `len=74`

> The fire brigades union say the practice of leaving firemen to deal with 999 calls is irresponsible and
> dangerous. Firemen with no medical training are being sent to life-threatening emergencies in South Yorkshire.
> Only a third of firefighters at the branch have any medical training, consisting of a four-day course of basic
> first aid. Fire engines are being turned into makeshift ambulances to ferry patients to A&E.

**Error category (LSTM):** Extractive echo, faithful (novel-bigram 0.00, unsup 0.00). Two source sentences copied; three of the reference's four points covered. No error.

**Error category (LLM):** Content-selection mismatch + over-length. Accurate throughout, but the South Yorkshire detail, the *only a third trained* figure and the four-day course all fall outside the reference. 74 tokens.


---

## 6. Most repetitive LSTM output

**Example id:** `b21cd549314996de79ba0ca2636c8ced66468267`, source length 1067 tokens

**Source (truncated for display):**

> los angeles (cnn)it's more than just one state's internal problem. the historic california drought hurts the
> rest of the union, too. that's because california is a breadbasket to the nation, growing more than a third of
> its vegetables and nearly two-thirds of its fruits and nuts. here's why we should heed the ongoing drought in
> the most populous state, a slowly expanding natural disaster now in its fourth year that this week prompted
> gov. jerry brown to announce a mandatory 25% cutback in water consumption in all cities. in 2014, one expert
> predicted consumers would pay more for some groceries because of the california drought. he was often right,
> according to statistics gathered by timothy richards, agribusiness professor at arizona state university.
> prices rose last year for these items on your kitchen table: . • berries rose in price by about 80 cents per
> clamshell to $3.88 . • brocco [...]

**Reference:**

> americans paid more for some fruits and vegetables last year because of the drought . tourists will now have
> to ask for a glass of water at a california restaurant . perhaps the only good thing is another "great" wine
> grape harvest last year .

**lstm_beam:**
`R1=23.3` `dup3=0.00` `novel2=0.25` `unsup=0.00` `oov=0.00` `len=56`

> new: " it's more than just one state's internal problem, " one expert says. " the drought will affect food
> prices again this year is unknown, thanks to a strong dollar... ". it is the most populous state, a slowly
> expanding natural disaster in the nation.

**llm_B_zeroshot:**
`R1=32.5` `dup3=0.01` `novel2=0.20` `unsup=0.07` `oov=0.02` `len=96`

> California is a breadbasket to the nation, growing more than a third of its vegetables and nearly two-thirds
> of its fruits and nuts. Gov. Jerry Brown announced a mandatory 25% cutback in water consumption in all cities
> due to the ongoing drought. Prices rose last year for berries, broccoli, grapes, melons, packaged salad, and
> peppers due to the drought. Fresh fruit prices are projected to rise between 2.5% and 3.5%, and vegetables
> between 2% and 3%, this year.

**Error category (LSTM):** **Grammatical breakage + quote misattribution.** *"it's more than just one state's internal problem," one expert says*, that is the article's own opening line, not an expert quote. The second sentence splices two source fragments into something ungrammatical; the third conflates California with the drought itself. unsup is 0.00 and dup3 is 0.00: every word is sourced and nothing repeats, yet the output is incoherent. **Neither diagnostic detects this.**

**Error category (LLM):** Content-selection mismatch. Accurate and well-formed; covers the produce-price point but neither the tourist-water nor the wine-harvest point the editor kept. 96 tokens.


---

## 7. Highest LSTM OOV rate

**Example id:** `38a207accfb31afc694f16531d23d2cda79ac85d`, source length 234 tokens

**Source (truncated for display):**

> "jake the dog and finn the human. the fun will never end. adventure time." so begins the dreamy theme song
> intro to the strangely addictive cartoon network tv show that's centered around psychedelic characters like
> the ice king, marceline the vampire queen and, of course, jake and finn. now, mega-fans of the hit show can
> experience "adventure time" in the skies. thai smile, a subsidiary of thailand flag carrier thai airways, on
> thursday unveiled colorful new livery featuring jake, finn and the beloved princess bubblegum sprawled across
> an airbus a320 at bangkok's suvarnabhumi international airport. the interior of the plane also has an
> adventure time theme, with overhead bins, head rests and even air sickness bags covered in the faces of
> characters from the show. airlines show off their new flying colors . the adventure time plane is the result
> of a partnership between thai airways subsi [...]

**Reference:**

> thai airways subsidiary thai smile features cartoon network paint job on a320 jet . overhead bins, head rests
> and air sick bags feature characters from cartoon network .

**lstm_beam:**
`R1=14.8` `dup3=0.00` `novel2=0.03` `unsup=0.00` `oov=0.00` `len=35`

> " jake the dog and finn the human. the fun will never end. adventure time ". the adventure time flight takes
> place on april 4, heading from bangkok to phuket.

**llm_B_zeroshot:**
`R1=35.0` `dup3=0.05` `novel2=0.28` `unsup=0.05` `oov=0.00` `len=110`

> Thai Smile, a subsidiary of Thai Airways, unveiled a colorful new livery featuring characters from the cartoon
> show "Adventure Time" on an Airbus A320 at Bangkok's Suvarnabhumi International Airport. The interior of the
> plane also has an "Adventure Time" theme, with overhead bins, head rests, and air sickness bags covered in the
> faces of characters from the show. The airline's partnership with Cartoon Network Amazon, a water park near
> Pattaya, resulted in the themed plane. The inaugural Thai Smile "Adventure Time" flight takes place on April
> 4, heading from Bangkok to Phuket.

**Error category (LSTM):** Lead-bias miss. The output is the cartoon's theme-song epigraph quoted verbatim, plus one scheduling detail. The actual story, Thai Smile's Adventure Time livery on an A320, is absent, as are all four reference entities. Selected as *highest LSTM OOV rate*, yet that rate is 0.00: `<unk>` suppression means the rare-word failure surfaces as omission instead.

**Error category (LLM):** **Format drift, the clearest case in the set.** Accurate and complete, but 110 tokens against a 58-token reference, nearly double. The LLM's characteristic failure here is not faithfulness but failing to infer an unstated length specification.


---

## 8. Most unsupported LLM content (hallucination candidate)

**Example id:** `d40bdad8e09e3dee871af08ac328a08f0af53d44`, source length 390 tokens

**Source (truncated for display):**

> many people have experienced the loss of a childhood pet, but for one little boy, a funeral for his goldfish
> was almost too much to bear. the pet in question? a goldfish, named top. brooke geherman, from alberta,
> canada, posted a video of her young son, kowen, sending off his beloved pet the right way: by flushing it down
> the toilet. young kowen is devastated by the loss of his pet goldfish, top, and holds a toilet funeral . the
> home video is filmed by mother, brooke geherman, and sees kowen stood in the family bathroom, cradling his
> deceased goldfish. she asks her son: 'who's top?' kowen responds by pointing and responds solemnly: 'this
> one.' the young boy cradles the fish before kissing it repeatedly in a heartbreaking move . in a heartbreaking
> move, the young boy then proceeds to kiss the lifeless goldfish repeatedly. canadian mum brooke then asks:
> 'where are you going to put him?' he [...]

**Reference:**

> brooke geherman posted video of her son, kowen on youtube . young boy, from alberta, canada, cradles his
> deceased goldfish, top . kowen performs funeral by flushing goldfish before bursting into tears .

**lstm_beam:**
`R1=52.6` `dup3=0.00` `novel2=0.09` `unsup=0.00` `oov=0.00` `len=34`

> brooke, from alberta, canada, posted a video of her young son. he is devastated by the loss of his pet
> goldfish, top, and holds a toilet funeral.

**llm_B_zeroshot:**
`R1=42.0` `dup3=0.01` `novel2=0.48` `unsup=0.23` `oov=0.05` `len=85`

> A Canadian mother, Brooke Geherman, filmed her young son, Kowen, holding a toilet funeral for his deceased
> goldfish, Top. The video, captioned "boy loses best friend," has garnered 374,551 views on social media.
> Kowen, visibly distraught, cradled the lifeless fish and kissed it repeatedly before dropping it down the
> toilet. The boy's mother comforted him as he burst into tears, shrieking the name of his beloved pet, Top.

**Error category (LSTM):** Extractive echo, faithful. Drops the surname (*brooke*, not *brooke geherman*) and the closing detail that the boy burst into tears; otherwise accurate.

**Error category (LLM):** **Not hallucination, refuted by checking the source.** Selected as the highest unsupported-content output (0.23), but every specific is in the article: the *374,551* view count, the caption *boy loses best friend*, and *shriek* all occur verbatim. The score reflects paraphrase, not invention. This example is the evidence behind §5's refutation of the hallucination claim.


---

## 9. Longest article

**Example id:** `9c616b0acd51d58f0de5ad51a9efa7e38c47df4c`, source length 1976 tokens

**Source (truncated for display):**

> hundreds have attended the funeral of a nine-year-old calais schoolgirl who was raped and murdered by a polish
> migrant. the tragic killing of chloe sparked a national outpouring of grief in france after her body was
> discovered in a migrant camp last week. mourners gathered at st peter's in northern france today where huge
> screens had been erected to show the ceremony for the many who could not fit inside the packed church. scroll
> down for video . isabelle hyart lovingly touched the small white coffin of her daughter chloe as it was taken
> down to the waiting hearse . distraught: isabelle hyart breaks down in tears as she leaves the funeral of her
> daughter chloe . the procession of grieving family and friends follows the coffin of the nine-year-old outside
> st peter's church . pall bearers carry the coffin of chloe, who was raped and murdered on april 15, down the
> steps of the saint-pierre  [...]

**Reference:**

> mourners gathered outside st peter's to pay tribute to nine-year-old chloe . schoolgirl was snatched in front
> of her mother by zbigniew huminsk, 38 . she was raped and murdered and her body discovered in a migrant camp .

**lstm_beam:**
`R1=48.0` `dup3=0.00` `novel2=0.14` `unsup=0.00` `oov=0.00` `len=38`

> isabelle was found in a migrant camp last week. her body was found at st peter's in northern france. her
> daughter chloe had been snatched from a french playground in front of her mother isabelle.

**llm_B_zeroshot:**
`R1=43.4` `dup3=0.00` `novel2=0.30` `unsup=0.06` `oov=0.02` `len=94`

> Hundreds attended the funeral of nine-year-old Chloe in Calais, France, where she was raped and murdered by
> Polish migrant Zbigniew Huminsk. The tragic killing sparked a national outpouring of grief in France after her
> body was discovered in a migrant camp last week. Mourners gathered at St Peter's in northern France, with huge
> screens erected to show the ceremony for those who could not fit inside the packed church. Over 6,000 people
> protested against migrant crime on Saturday, following the appalling crime that shocked the country.

**Error category (LSTM):** **Relational error, the most serious LSTM failure in the set, and invisible to every diagnostic.** *"isabelle was found in a migrant camp ... her body was found at st peter's"*: Isabelle Hyart is the **living mother** who attended the funeral; the victim is her nine-year-old daughter Chloe, and St Peter's is the church, not the discovery site. unsup is 0.00 because every content word does appear in the article, **the words are sourced and the relations between them are invented.** Confirms that unsupported-content is a lexical proxy that cannot see this class of error.

**Error category (LLM):** Content-selection mismatch. Accurate, including the attacker's name; adds the 6,000-strong protest the reference omits. 94 tokens.


---

## 10. Shortest article

**Example id:** `417abb61e9c6f845ceb5bd73a63d479c9110d09a`, source length 59 tokens

**Source (truncated for display):**

> the build-up for the blockbuster fight between floyd mayweather and manny pacquiao in las vegas on may 2 steps
> up a gear on tuesday night when the american holds an open workout for the media. the session will be streamed
> live across the world and you can watch it here from 12am uk (7pm edt).

**Reference:**

> floyd mayweather holds an open media workout from 12am uk (7pm edt) . the american takes on manny pacquiao in
> las vegas on may 2 . mayweather's training is being streamed live across the world .

**lstm_beam:**
`R1=63.2` `dup3=0.00` `novel2=0.05` `unsup=0.00` `oov=0.00` `len=45`

> floyd mayweather and manny pacquiao in las vegas on may 2 steps up a gear on tuesday night. the american holds
> an open workout for the media. the session will be streamed live across the world and you can watch it here.

**llm_B_zeroshot:**
`R1=65.1` `dup3=0.00` `novel2=0.43` `unsup=0.23` `oov=0.02` `len=57`

> Floyd Mayweather is holding an open workout for the media on Tuesday night. The session will be streamed live
> across the world. The event is part of the build-up for the May 2 fight against Manny Pacquiao in Las Vegas.
> The workout will take place at 12am UK time (7pm EDT).

**Error category (LSTM):** Extractive echo with a truncation defect. Near-verbatim, but the opening drops *the build-up for the blockbuster fight between*, leaving *"floyd mayweather and manny pacquiao in las vegas on may 2 steps up a gear"*, ungrammatical. The content is correct.

**Error category (LLM):** Faithful and aligned, the best length match in the set at 57 tokens. Its unsup of 0.23 is a **denominator artefact**: the source article is only 55 words, so ordinary paraphrase moves the rate sharply. A caution against reading this proxy on short inputs.


---

## 11. Most abstractive LSTM output

**Example id:** `9ee69bc01d69e0b4b217ff8638ff4fc2f994b0b3`, source length 331 tokens

**Source (truncated for display):**

> it takes a lot to impress barcelona's all-star cast of players, but young defender martin montoya provoked
> gasps from his team-mates with an outrageous trick shot during training. the young spanish defender made his
> hoop dreams come true by netting a lofted pass from around 30 yards 'downtown' straight into a basketball ring
> on the other side of the club's ciutat esportiva training ground. after the ball drops into the net, a group
> of watching players applaud while others, including ivan rakitic and javier mascherano, race over to
> congratulate the 23-year-old left-back. martin montoya (centre) lines up a lofted pass during the training
> session at ciutat esportiva . the ball sails high in the air seemingly in the direction of no one while
> players look on . incredibly, the ball sinks directly into a basketball hoop at the other side of the training
> pitch . the young spanish defender gives  [...]

**Reference:**

> martin montoya netted a 30yard pass into a basketball hoop at training . barcelona maintained four-point in la
> liga lead with victory over almeria . luis suarez and lionel messi scored stunning curled strikes at nou camp
> .

**lstm_beam:**
`R1=12.1` `dup3=0.00` `novel2=0.43` `unsup=0.15` `oov=0.00` `len=31`

> real madrid beat rayo vallecano on wednesday. real madrid played after a 2-0 defeat of rayo vallecano.
> barcelona beat almeria on wednesday after a 1-0 defeat of atletico.

**llm_B_zeroshot:**
`R1=28.8` `dup3=0.00` `novel2=0.37` `unsup=0.10` `oov=0.01` `len=87`

> Martin Montoya, a 23-year-old left-back, made a trick shot during training by netting a lofted pass from 30
> yards into a basketball ring. The ball sailed high in the air, seemingly in the direction of no one, before
> sinking directly into the hoop. A group of watching players, including Ivan Rakitic and Javier Mascherano,
> applauded and rushed to congratulate Montoya. Barcelona's relentless march towards reclaiming the La Liga
> title moved closer with a 4-0 hammering of Almeria.

**Error category (LSTM):** **Topic miss + fabrication, verified.** It summarises the league-results footer instead of the story the reference leads on (Montoya's trick shot). *atletico* does not occur in the article, and Barcelona beat Almeria **4-0, not 1-0**; Real Madrid's 2-0 over Rayo Vallecano is correctly copied. Fluent, confident and wrong, **with attention present**, which qualifies §5's attribution of this mode to the no-attention ablation.

**Error category (LLM):** Faithful and aligned. Trick shot, distance, both named team-mates and the Almeria scoreline all correct. Misses the reference's Suarez/Messi point. 87 tokens.


---

## 12. Longest LLM output (format adherence)

**Example id:** `3e4b0bd8f208707de67f324573bbb95f269d0a33`, source length 516 tokens

**Source (truncated for display):**

> many couples disagree over what to name their children, but one new york couple’s battle over their unborn
> baby’s moniker became so unbearable that they have asked random members of the public to help them reach a
> decision. kseniya soukeras, who is due in august, wants to name her unborn child michael, after her late
> father. her husband nicholas prefers the greek name spyridon, after his own father. because they have been
> unable to settle their disagreement privately, the couple has turned to the general public to help them make a
> decision; nicholas, 37, has launched an online petition asking for help to sway his wife. baby name showdown:
> nicholas and kseniya soukeras can't agree over whether to name their child spyridon or michael, which nicholas
> says his wife will shorten to mischa . 'the argument is serious, it’s not a joke,' kseniya, 33, told the new
> york post. 'i don’t want to cal [...]

**Reference:**

> nicholas soukeras, 37, of queens, new york wants his future son to be called spyridon, after his father . his
> wife, kseniya, 33, doesn't like the 'archaic' name and prefers to call their child michael . nicholas needs
> his online petition to earn 100,000 signatures for his wife to relent . ironically, the couple don’t actually
> know for certain that they are having a boy .

**lstm_beam:**
`R1=32.1` `dup3=0.00` `novel2=0.03` `unsup=0.00` `oov=0.00` `len=73`

> new york couple ’ s battle over their unborn baby ’ s moniker became so unbearable that they have asked random
> members of the public to help them reach a decision. nicholas, 37, has launched an online petition asking for
> help to sway his wife. he has been held by the highest and most honored officials and patriots of both the
> u.s. and the former soviet union.

**llm_B_zeroshot:**
`R1=43.8` `dup3=0.01` `novel2=0.21` `unsup=0.03` `oov=0.05` `len=118`

> Kseniya Soukeras, 33, wants to name her unborn child Michael, after her late father, while her husband
> Nicholas prefers the Greek name Spyridon, after his own father. The couple has turned to the public to help
> them make a decision, with Nicholas launching an online petition asking for help to sway his wife. Kseniya
> finds the name Spyridon 'archaic, base, bereft of elegance and unsuitable for a child living in 21st century
> America'. The couple does not even know for sure whether they are having a son, with Kseniya joking that her
> husband thinks he's a psychic and can see through her belly.

**Error category (LSTM):** Topic drift. Two sentences copied accurately, then a non-sequitur, *"he has been held by the highest and most honored officials and patriots of both the u.s. and the former soviet union"*, lifted from elsewhere in the article and attached to the wrong subject. unsup 0.00: sourced text, wrong place.

**Error category (LLM):** **Format drift, the longest output in the set at 118 tokens**, against a ~70-token reference, despite variant B's explicit *3-4 sentences, ~55 words*. Content is accurate and covers three of the four reference points; the failure is specification adherence, not truth.


---

## 13. Shortest LSTM output

**Example id:** `1030d228c1c1b10cc365c8d3ae6293ddbb9d0dda`, source length 275 tokens

**Source (truncated for display):**

> efe obada has been signed by the dallas cowboys, despite playing just five games for the london warriors. the
> 22-year-old londoner, who currently works in a warehouse in welwyn garden city, will fly out to texas next
> month for rookie mini-camp. obada only started playing american football half way through last season and was
> mentored by warriors defensive co-ordinator aden durde. efe obada joins the dallas cowboys next month after a
> whirlwind year and just five american football games . the 22-year-old only took up the game last year and
> finds himself on an nfl roster in a remarkable tale . spotted by warriors defensive co-ordinator aden durde,
> obada has worked for the last six months . the 6ft 5in, 225-pound obada is seen as a defensive end or a tight
> end and will fly to texas next month . durde, a former nfl linebacker, was a coaching intern with the cowboys
> last summer and spent six m [...]

**Reference:**

> obada has been signed by the cowboys despite playing just five games with the britbowl champions . the
> 22-year-old londoner only started playing american football last year . he has been mentored by warriors
> defensive co-ordinator aden durde, who spent last summer as a coaching intern with the cowboys . obada will
> give up his job in a welwyn garden city factory and head to rookie mini-camp in texas in may .

**lstm_beam:**
`R1=36.4` `dup3=0.00` `novel2=0.08` `unsup=0.00` `oov=0.00` `len=25`

> the 22-year-old has been signed by the dallas cowboys next month. the 22-year-old only started playing
> american football half way through last season.

**llm_B_zeroshot:**
`R1=55.9` `dup3=0.00` `novel2=0.15` `unsup=0.05` `oov=0.08` `len=66`

> Efe Obada has been signed by the Dallas Cowboys, despite playing just five games for the London Warriors. The
> 22-year-old Londoner will fly out to Texas next month for rookie mini-camp. Obada was mentored by Warriors
> defensive co-ordinator Aden Durde, who spotted him and worked with him for six months. Efe Obada joins
> compatriot Jack Crawford on the Cowboys roster.

**Error category (LSTM):** Entity omission + temporal conflation. In 25 tokens, the shortest output in the set, it never names Efe Obada, and *"has been signed by the dallas cowboys next month"* merges a completed signing with next month's mini-camp. Both reference-critical entities (Obada, Aden Durde) are missing.

**Error category (LLM):** Faithful and aligned. Names every entity; *jack crawford* and *compatriot* are both verified present in the source. 66 tokens, close to the reference.

