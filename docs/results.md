# Results

## Protocol

| Setting | Value |
|:---:|:---:|
| Benchmark | SWE-bench Verified |
| Tasks | 500 |
| Rollouts per task | 1 |
| Temperature | 0 |
| Top-p | 1.0 |
| Maximum model length | 32,768 |
| Maximum tool-call turns | 40 |
| Dataset revision | `91aa3ed51b709be6457e12d00300a6a596d4c6a3` |
| Harness revision | `f7bbbb2ccdf479001d6467c9e34af59e44a840f9` |

## Runs

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center">Model</th>
      <th rowspan="2" align="center">Score</th>
      <th colspan="7" align="center">Outcomes</th>
      <th colspan="2" align="center">Diagnostics</th>
    </tr>
    <tr>
      <th align="center">Resolved</th>
      <th align="center">Submitted</th>
      <th align="center">Overlong</th>
      <th align="center">Capped</th>
      <th align="center">Malformed</th>
      <th align="center">Infra</th>
      <th align="center">Total</th>
      <th align="center">Empty</th>
      <th align="center">Errors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">SFT base</td>
      <td align="center">11.0%</td>
      <td align="center">55</td>
      <td align="center">205</td>
      <td align="center">136</td>
      <td align="center">103</td>
      <td align="center">1</td>
      <td align="center">0</td>
      <td align="center">500</td>
      <td align="center">120</td>
      <td align="center">2</td>
    </tr>
    <tr>
      <td align="center">Dr. GRPO<br>Stage 1 · ckpt-20</td>
      <td align="center">12.4%</td>
      <td align="center">62</td>
      <td align="center">196</td>
      <td align="center">139</td>
      <td align="center">100</td>
      <td align="center">0</td>
      <td align="center">3</td>
      <td align="center">500</td>
      <td align="center">128</td>
      <td align="center">1</td>
    </tr>
  </tbody>
</table>
