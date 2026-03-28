pipeline {
    agent any

    triggers {
        // Poll every 5 minutes; webhook can override this
        pollSCM('H/5 * * * *')
    }

    environment {
        IMAGE_NAME = 'ghcr.io/obtuse-triangle/trustops-gateway'
        IMAGE_TAG  = "${GIT_COMMIT}"
    }

    stages {
        stage('Setup') {
            steps {
                checkout scm
                sh '''
                    python3 -m venv .venv
                    .venv/bin/pip install --upgrade pip --quiet
                    .venv/bin/pip install -r requirements.txt --quiet
                    .venv/bin/pip install ruff --quiet
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '.venv/bin/ruff check app/ eval/'
            }
        }

        stage('Test') {
            steps {
                sh '.venv/bin/pytest tests/ -v --tb=short --junitxml=test-results.xml'
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }

        stage('Build') {
            steps {
                sh 'docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .'
            }
        }

        stage('Push') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'ghcr-credentials',
                    usernameVariable: 'GHCR_USER',
                    passwordVariable: 'GHCR_TOKEN'
                )]) {
                    sh '''
                        echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
                        docker push ${IMAGE_NAME}:${IMAGE_TAG}
                    '''
                }
            }
        }

        stage('Tag Latest') {
            steps {
                sh '''
                    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                    docker push ${IMAGE_NAME}:latest
                '''
            }
        }
    }

    post {
        always {
            echo "Pipeline finished for commit ${GIT_COMMIT}"
        }
        failure {
            echo "BUILD FAILED — ${IMAGE_NAME}:${IMAGE_TAG} was NOT pushed"
        }
        success {
            echo "SUCCESS — pushed ${IMAGE_NAME}:${IMAGE_TAG} and ${IMAGE_NAME}:latest"
        }
    }
}
