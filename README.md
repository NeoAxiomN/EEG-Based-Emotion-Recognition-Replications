


# 🚀 EEGLearning
Under continuous update...

This repository is my personal collection for replicating and exploring various methodologies in EEG-based emotion recognition. My goal is to systematically reproduce key findings from published research, compare my results with reported figures, and provide a clear, accessible overview of different approaches in the field.

# 📈 Replication Results
The following table summarizes my replication efforts, comparing my achieved results with those reported in the original papers.

<table>
  <thead>
    <tr>
      <th colspan="1">Method</th>
      <th colspan="4">Experimental Settings</th>
      <th colspan="3">Results</th>
    </tr>
    <tr>
      <th>Method</th>
      <th>Dataset</th>
      <th>Task</th>
      <th>Evaluation</th>
      <th>Validation Method</th>
      <th>Reported (%)</th>
      <th>Replication (%)</th>
      <th>Gap (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">ACRNN</td>
      <td rowspan="2">DEAP</td>
      <td>Arousal</td>
      <td>ACC</td>
      <td rowspan="2">Subject-Dependent<br>10-Fold Cross-Validation</td>
      <td>93.38±3.73</td>
      <td>94.22±3.95</td>
      <td>+0.84↑</td>
    </tr>
    <tr>
      <td>Valence</td>
      <td>ACC</td>
      <td>93.72±3.21</td>
      <td>92.53±5.32</td>
      <td>-1.19↓</td>
    </tr>
    <tr>
      <td rowspan="6">HSLT</td>
      <td rowspan="6">DEAP</td>
      <td rowspan="2">Arousal</td>
      <td>ACC</td>
      <td rowspan="6">Subject-Independent<br>Leave-One-Subject-Out Cross-Validation</td>
      <td>65.75±8.51</td>
      <td>64.34±13.45</td>
      <td>-1.41↓</td>
    </tr>
    <tr>
      <td>F1</td>
      <td>64.29±10.06</td>
      <td>60.20±13.86</td>
      <td>-4.09↓</td>
    </tr>
    <tr>
      <td rowspan="2">Valence</td>
      <td>ACC</td>
      <td>66.51±8.53</td>
      <td>62.91±9.11</td>
      <td>-3.60↓</td>
    </tr>
    <tr>
      <td>F1</td>
      <td>66.27±7.29</td>
      <td>57.99±10.34</td>
      <td>-8.28↓</td>
    </tr>
    <tr>
      <td rowspan="2">4-Classes</td>
      <td>ACC</td>
      <td>56.93±8.22</td>
      <td>44.97±15.22</td>
      <td>-11.96↓</td>
    </tr>
    <tr>
      <td>F1</td>
      <td>54.29±8.59</td>
      <td>37.69±15.35</td>
      <td>-16.60↓</td>
    </tr>
  </tbody>
</table>

# 📧 Contact
For any questions or discussions about this work, feel free to reach out to me.