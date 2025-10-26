# AudioProcessor Kubernetes Deployment

Этот каталог содержит все необходимые манифесты Kubernetes для развертывания AudioProcessor в продакшн среде.

## 📁 Структура файлов

```
k8s/
├── namespace.yaml              # Namespaces для ml-service и monitoring
├── rbac.yaml                  # RBAC настройки и ServiceAccount
├── configmap.yaml             # ConfigMaps для конфигурации
├── secret.yaml                # Secrets для чувствительных данных
├── services.yaml              # Kubernetes Services
├── audio-processor-deployment.yaml  # Deployment для API
├── audio-worker-deployment.yaml     # Deployment для Workers
├── ingress.yaml               # Ingress для внешнего доступа
├── hpa.yaml                   # Horizontal Pod Autoscaler
├── monitoring.yaml            # Prometheus, Grafana, AlertManager
├── kustomization.yaml         # Kustomize конфигурация
├── deploy.sh                  # Скрипт автоматического развертывания
└── README.md                  # Этот файл
```

## 🚀 Быстрое развертывание

### Автоматическое развертывание

```bash
# Запуск скрипта развертывания
./deploy.sh
```

### Ручное развертывание

```bash
# 1. Создание namespaces
kubectl apply -f namespace.yaml

# 2. Настройка RBAC
kubectl apply -f rbac.yaml

# 3. Применение конфигурации
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml  # Обновите секреты перед применением!

# 4. Создание сервисов
kubectl apply -f services.yaml

# 5. Развертывание приложений
kubectl apply -f audio-processor-deployment.yaml
kubectl apply -f audio-worker-deployment.yaml

# 6. Настройка мониторинга
kubectl apply -f monitoring.yaml

# 7. Настройка автомасштабирования
kubectl apply -f hpa.yaml

# 8. Настройка Ingress
kubectl apply -f ingress.yaml
```

## ⚙️ Конфигурация

### Обновление секретов

Перед развертыванием обновите `secret.yaml` с вашими реальными значениями:

```bash
# Создание секретов через kubectl
kubectl create secret generic ml-service-secrets \
  --from-literal=S3_ACCESS_KEY=your-access-key \
  --from-literal=S3_SECRET_KEY=your-secret-key \
  --from-literal=MASTERML_TOKEN=your-token \
  --from-literal=DATABASE_PASSWORD=your-password \
  --from-literal=HUGGINGFACE_TOKEN=your-huggingface-token \
  --namespace=ml-service

kubectl create secret generic monitoring-secrets \
  --from-literal=GRAFANA_ADMIN_USER=admin \
  --from-literal=GRAFANA_ADMIN_PASSWORD=your-secure-password \
  --from-literal=SLACK_WEBHOOK_URL=your-slack-webhook \
  --namespace=monitoring
```

### Настройка доменов

Обновите домены в `ingress.yaml`:

```yaml
# Замените example.com на ваш домен
- host: ml-service.yourdomain.com
- host: monitoring.yourdomain.com
```

### Настройка образов

Обновите образы в deployment файлах:

```yaml
# Замените на ваш registry
image: your-registry.com/audio-processor:latest
image: your-registry.com/audio-processor:gpu
```

## 📊 Мониторинг

После развертывания доступны следующие сервисы:

- **AudioProcessor API**: `http://ml-service.yourdomain.com/audio`
- **Flower (Celery)**: `http://ml-service.yourdomain.com/flower`
- **Prometheus**: `http://monitoring.yourdomain.com/prometheus`
- **Grafana**: `http://monitoring.yourdomain.com/grafana` (admin/admin)
- **AlertManager**: `http://monitoring.yourdomain.com/alertmanager`

## 🔧 Управление

### Проверка статуса

```bash
# Статус подов
kubectl get pods -n ml-service
kubectl get pods -n monitoring

# Статус сервисов
kubectl get svc -n ml-service
kubectl get svc -n monitoring

# Статус HPA
kubectl get hpa -n ml-service
```

### Масштабирование

```bash
# Ручное масштабирование
kubectl scale deployment audio-worker --replicas=10 -n ml-service

# Проверка HPA
kubectl get hpa -n ml-service
kubectl describe hpa audio-worker-hpa -n ml-service
```

### Логи

```bash
# Логи API
kubectl logs -f deployment/audio-processor -n ml-service

# Логи workers
kubectl logs -f deployment/audio-worker -n ml-service

# Логи GPU workers
kubectl logs -f deployment/audio-gpu-worker -n ml-service
```

### Отладка

```bash
# Описание пода
kubectl describe pod <pod-name> -n ml-service

# Выполнение команд в поде
kubectl exec -it deployment/audio-processor -n ml-service -- /bin/bash

# Проверка событий
kubectl get events -n ml-service --sort-by=.metadata.creationTimestamp
```

## 🚨 Алерты

Настроены следующие алерты:

### Критические
- Высокий уровень ошибок (>5%)
- Очередь задач переполнена (>1000)
- Высокое использование GPU памяти (>90%)
- Сервис недоступен

### Предупреждения
- Высокое использование CPU (>80%)
- Высокое использование памяти (>80%)
- Медленная обработка (>5 минут)
- Низкий уровень успешности extractors (<95%)

## 🔄 Обновление

### Обновление образа

```bash
# Обновление образа
kubectl set image deployment/audio-processor audio-processor=your-registry.com/audio-processor:v1.1.0 -n ml-service
kubectl set image deployment/audio-worker audio-worker=your-registry.com/audio-processor:v1.1.0 -n ml-service

# Проверка статуса обновления
kubectl rollout status deployment/audio-processor -n ml-service
```

### Откат

```bash
# Откат к предыдущей версии
kubectl rollout undo deployment/audio-processor -n ml-service
kubectl rollout undo deployment/audio-worker -n ml-service
```

## 🧹 Очистка

```bash
# Удаление всех ресурсов
kubectl delete -f .
kubectl delete namespace ml-service
kubectl delete namespace monitoring
```

## 📝 Примечания

1. **GPU Workers**: Требуют ноды с GPU и NVIDIA device plugin
2. **Persistent Storage**: Для продакшна рекомендуется использовать PersistentVolumes
3. **SSL/TLS**: Настройте cert-manager для автоматических сертификатов
4. **Backup**: Настройте регулярное резервное копирование данных
5. **Security**: Регулярно обновляйте образы и зависимости

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте логи: `kubectl logs -f deployment/audio-processor -n ml-service`
2. Проверьте события: `kubectl get events -n ml-service`
3. Проверьте ресурсы: `kubectl top pods -n ml-service`
4. Обратитесь к команде разработки
