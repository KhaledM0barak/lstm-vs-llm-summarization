# Demo articles

Out-of-domain text for `python -m src.demo`, used in the video walkthrough.

`demo_article_battery.txt`, a battery-chemistry news article. Chosen because its
vocabulary sits outside the model's 50,000-type training vocabulary:
`electrolyte`, `anode`, `graphite`, `fluorinated` and the researcher's name are
all unreachable. Token-level OOV rate **5.3%**, against **1.83%** on in-domain
training text.

The LSTM's output ends the first clause at "have developed a battery", it cannot
emit `electrolyte`, then skips every technical finding for the one sentence it
can express. A live demonstration of the OOV failure described in §5 of the
report.

```bash
python -m src.demo --file examples/demo_article_battery.txt
```
