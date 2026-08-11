# English/Japanese/Korean ASR benchmark

- Clip duration: 60 seconds
- Pipeline mode: `accurate`
- Model loading and a 10-second warm-up are excluded from timed results.

| Language | Model | Audio | Elapsed | Speed | Segments | Characters | Words |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en | `large-v3` | 60.0s | 4.02s | 14.94x | 15 | 1093 | 189 |
| ja | `large-v3` | 60.0s | 2.72s | 22.03x | 13 | 163 | 127 |
| ko | `large-v3` | 60.0s | 2.84s | 21.12x | 8 | 264 | 62 |
| en | `turbo` | 60.0s | 1.53s | 39.29x | 15 | 1081 | 188 |
| ja | `turbo` | 60.0s | 2.06s | 29.08x | 12 | 170 | 132 |
| ko | `turbo` | 60.0s | 1.19s | 50.63x | 9 | 277 | 70 |

## `turbo` compared with `large-v3`

| Language | Speed-up | Segment delta | Character delta | Transcript agreement |
| --- | ---: | ---: | ---: | ---: |
| en | 2.63x | +0 | -12 | 99.0% |
| ja | 1.32x | -1 | +7 | 90.7% |
| ko | 2.40x | +1 | +13 | 92.5% |

> Transcript agreement only measures how similar the two model outputs are. Use the generated SRT files for human accuracy review.
