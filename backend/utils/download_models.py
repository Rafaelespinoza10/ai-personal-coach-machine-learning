import boto3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

env_path = project_root / '.env'
local_env_path = project_root / '.env.local'

load_dotenv(env_path)
load_dotenv(local_env_path, override=True)

def download_models():
    bucket = os.getenv('AWS_BUCKET_NAME')
    if not bucket:
        print('ERROR: AWS_BUCKET_NAME no está configurado en las variables de entorno', file=sys.stderr)
        sys.exit(1)
    
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
    
    try:
        s3 = boto3.client(
            's3',
            region_name=region,
            config=boto3.session.Config(
                retries={'max_attempts': 2, 'mode': 'standard'},
                max_pool_connections=10,
            )
        )
    except Exception as e:
        print(f'ERROR: No se pudo crear cliente S3: {e}', file=sys.stderr)
        sys.exit(1)

    models = [
        # Sentiment
        'sentiment/model_explainer.pkl',
        # Mental Health
        'mental_health/best_mental_health_model.pkl',
        'mental_health/scaler.pkl',
        'mental_health/selected_features.pkl',
        'mental_health/preprocessors.pkl',
        'mental_health/model_config.json',
        # Stress (main)
        'best_model.pkl',
        'scaler.pkl',
        'preprocessors.pkl',
        'training_results.json',
    ]

    if os.path.exists('/app/models'):
        base_path = '/app/models'
    else:
        base_path = str(project_root / 'models')
    
    print(f'Using models path: {base_path}')
    
    # Estadísticas para monitoreo
    downloaded_count = 0
    skipped_count = 0
    total_size = 0
    
    for model_key in models:
        s3_key = f'models/{model_key}'
        local_path = os.path.join(base_path, model_key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path):
            try:
                response = s3.head_object(Bucket=bucket, Key=s3_key)
                remote_size = response.get('ContentLength', 0)
                local_size = os.path.getsize(local_path)
                
                if local_size == remote_size:
                    print(f'✓ Model {model_key} already exists and matches S3 ({local_size} bytes)')
                    skipped_count += 1
                    continue
                else:
                    print(f'Model {model_key} size mismatch (local: {local_size}, S3: {remote_size}), re-downloading...')
            except Exception as e:
                print(f'⚠ Could not verify {model_key}, re-downloading: {e}')
        
        try:
            print(f'Downloading {model_key} from S3...')
            s3.download_file(bucket, s3_key, local_path)
            
            file_size = os.path.getsize(local_path)
            total_size += file_size
            downloaded_count += 1
            print(f'✓ Downloaded {model_key} ({file_size:,} bytes)')
        except Exception as e:
            print(f'ERROR: No se pudo descargar {model_key}: {e}', file=sys.stderr)
            sys.exit(1)
    
    print('\n' + '='*50)
    print('Download Summary:')
    print(f'  Downloaded: {downloaded_count} files')
    print(f'  Skipped: {skipped_count} files')
    print(f'  Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)')
    print('='*50)
    print('Model download process completed')
            
if __name__ == '__main__':
    download_models()