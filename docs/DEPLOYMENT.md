# Despliegue en Hetzner (Ciclo 4)

> **Objetivo:** levantar la app de segmentación vertebral en un servidor
> Hetzner público, con HTTPS automático y pesos cargados desde Hugging Face
> Hub. El jurado abre una URL y la usa.

## 1. Pre-requisitos en el servidor

| Requisito | Cómo verificar | Cómo instalar (Ubuntu/Debian) |
|-----------|----------------|--------------------------------|
| Docker + Docker Compose v2 | `docker compose version` | `curl -fsSL https://get.docker.com \| sh && sudo usermod -aG docker $USER` (volver a entrar) |
| Git | `git --version` | `sudo apt update && sudo apt install -y git` |
| Puertos 80 y 443 abiertos | `sudo ufw status` o panel Hetzner | `sudo ufw allow 80,443/tcp` |
| Al menos 4 GB RAM y 10 GB libres en disco | `free -h && df -h` | n/a |
| IP pública fija | desde el panel Hetzner | (la asignan al crear el VPS) |

## 2. Pre-requisitos en Hugging Face Hub (una sola vez)

Sigue [`HUGGINGFACE_SETUP.md`](HUGGINGFACE_SETUP.md):
1. Crear cuenta en https://huggingface.co
2. Generar un Access Token tipo **Write**
3. Subir los dos `.pth` (DeepLabV3+ multiclase + UNet binario) al repo
   `<tu-usuario>/spine-checkpoints` con `scripts/upload_weights.py`

Anota el `repo_id` resultante — lo necesitas en el paso 4.

## 3. Deploy automatizado (camino feliz)

```bash
# En el server, en el directorio donde quieras instalar la app:
git clone https://github.com/ElvLandau117/Segmentacion-Soluciones.git spine-segmentation
cd spine-segmentation

# Copiar el template de env y editarlo con tus valores
cp .env.example .env
nano .env
#   -> ajustar HF_REPO_ID y DOMAIN como mínimo
#   -> HF_TOKEN solo si el repo de HF es privado

# Ejecutar el script de deploy
bash scripts/deploy_hetzner.sh
```

El script:
1. Valida que `.env` exista y tenga `HF_REPO_ID` y `DOMAIN`.
2. Construye la imagen Docker (`docker compose build`).
3. Levanta el stack (`docker compose up -d`).
4. Espera al healthcheck y muestra el estado final.

Tiempo total: **~10–15 min** (build + descarga de pesos en el primer arranque).

## 4. Deploy manual (paso a paso)

Si prefieres entender qué pasa o si el script falla, hazlo a mano:

```bash
# 1) Construir la imagen (multi-stage, ~3 GB en disco)
docker compose build

# 2) Arrancar el stack en background
docker compose up -d

# 3) Ver logs del primer arranque (la descarga de HF toma 30-60 s)
docker compose logs -f app

# 4) Cuando veas "Running on local URL: http://0.0.0.0:7860", abrir:
#    https://<TU_DOMAIN>
```

## 5. Verificación end-to-end (acceptance)

| Check | Comando / acción |
|-------|------------------|
| Containers up | `docker compose ps` → `app` y `caddy` en estado **healthy** |
| HTTPS responde | `curl -I https://<DOMAIN>/health` → `200 OK` |
| Certificado válido | el browser muestra candadito verde (Let's Encrypt) |
| App carga | abrir `https://<DOMAIN>` → ver la UI con 4 tabs |
| Disclaimer visible | scroll abajo → "Aviso medico..." al pie |
| Inferencia funciona | subir una radiografía del test set → Cobb + segmentación aparece |
| Latencia razonable | medir tiempo entre "Analyze" y resultado (objetivo: < 8 s) |
| Caché persiste | `docker compose restart app` → no re-descarga los .pth |

## 6. Operaciones comunes

| Acción | Comando |
|--------|---------|
| Ver logs | `docker compose logs -f app` |
| Reiniciar la app | `docker compose restart app` |
| Subir un nuevo peso | (1) subir a HF con `scripts/upload_weights.py`, (2) `docker compose restart app` |
| Pull de cambios del repo | `git pull && docker compose up -d --build` |
| Cambiar el dominio | editar `.env`, `docker compose up -d` |
| Apagar todo (preservar cache) | `docker compose down` |
| Apagar y resetear cache de pesos | `docker compose down -v` (re-descarga al levantar) |
| Ver uso de disco/memoria | `docker stats` |
| Renovación de SSL | Caddy lo hace automáticamente cada ~60 días |

## 7. Troubleshooting

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| `caddy` no arranca, error `tls: handshake failure` | DNS no apunta al server, o puerto 80 cerrado | Verificar `dig $DOMAIN`, abrir puerto 80 en firewall |
| `app` queda en `starting` > 5 min | Descarga de HF cuelga o falla | `docker compose logs app`; verificar `HF_REPO_ID` y conectividad a `huggingface.co` |
| Inferencia tarda > 30 s | CPU del Hetzner pequeño | Considerar tier con más CPU, o bajar `INFERENCE_IMAGE_SIZE` a 384 |
| `Let's Encrypt rate limit` | Demasiados intentos en poco tiempo | Esperar 1 h, o usar staging: `acme_ca https://acme-staging-v02.api.letsencrypt.org/directory` |
| App responde 502 Bad Gateway | `app` no healthy | `docker compose ps`; `docker compose logs app` |
| Falta espacio en disco | Pesos + imagen + caché Docker | `docker system prune -a --volumes` (cuidado: elimina caché) |

## 8. Datos del despliegue actual

> Esta sección la actualiza Elvis después de hacer el deploy real.

- **Servidor:** Hetzner CPX21 (o el tier que termine usándose)
- **IP pública:** `<por completar>`
- **DOMAIN:** `<por completar>`.nip.io
- **HF_REPO_ID:** `<por completar>/spine-checkpoints`
- **URL pública:** `https://<por completar>`
- **Fecha de primer deploy:** `<por completar>`
- **Tamaño imagen Docker:** `<por completar>`
- **Latencia media por inferencia (CPU):** `<por completar>` s
- **Caché de pesos:** `<por completar>` MB
