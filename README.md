# EEGLearning
Just giving it a try.

<style>
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px auto;
    font-family: sans-serif;
  }
  th, td {
    border: 1px solid #dddddd;
    text-align: center;
    padding: 8px;
  }
  th {
    background-color: #f2f2f2;
    font-weight: bold;
  }
  tr:nth-child(even) {
    background-color: #f9f9f9;
  }
  .main-header {
    background-color: #e0e0e0;
    font-size: 1.1em;
    padding: 10px;
  }
</style>

<table>
  <thead>
    <tr>
      <th class="main-header" colspan="1">Method</th>
      <th class="main-header" colspan="4">Experimental Settings</th>
      <th class="main-header" colspan="3">Results</th>
    </tr>
    <tr>
      <th>Method</th>
      <th>Dataset</th>
      <th>Task</th>
      <th>Evaluation</th>
      <th>Validation Method</th>
      <th>Reported (%)</th>
      <th>Ours (%)</th>
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
      <td>93.38 ± 3.73</td>
      <td>94.22 ± 3.95</td>
      <td>+0.84↑</td>
    </tr>
    <tr>
      <td>Valence</td>
      <td>ACC</td>
      <td>93.72 ± 3.21</td>
      <td>92.53 ± 5.32</td>
      <td>-1.19↓</td>
    </tr>
  </tbody>
</table>

