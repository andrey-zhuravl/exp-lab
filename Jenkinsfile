pipeline {
  agent any
  options { timestamps() }
  stages {
    stage('Setup Python') { steps { sh 'python3 -V || python -V' } }
    stage('Install') { steps { sh 'pip install -e .' } }
    stage('Run A1') { steps { sh 'exp run -m experiments/a1.quickstart.yaml' } }
  }
  post {
    always { archiveArtifacts artifacts: 'out/a1/report.md', onlyIfSuccessful: false }
  }
}
