#!/bin/bash

# Docker Testing Script for AudioProcessor
# Usage: ./test_docker.sh [cpu|gpu]

set -e

COMPOSE_FILE="docker-compose.yml"
COMPOSE_GPU_FILE="docker-compose.gpu.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if NVIDIA Docker is available (for GPU)
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_error "NVIDIA Docker is not available. Please install nvidia-docker2 and try again."
        exit 1
    fi
    print_success "NVIDIA Docker is available"
}

# Function to build and start services
start_services() {
    local compose_file=$1
    local service_name=$2
    
    print_status "Building and starting $service_name services..."
    
    # Stop any existing services
    docker-compose -f $compose_file down > /dev/null 2>&1 || true
    
    # Build and start services
    docker-compose -f $compose_file up --build -d
    
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check if services are running
    if docker-compose -f $compose_file ps | grep -q "Up"; then
        print_success "$service_name services are running"
    else
        print_error "Failed to start $service_name services"
        docker-compose -f $compose_file logs
        exit 1
    fi
}

# Function to run tests
run_tests() {
    local compose_file=$1
    local service_name=$2
    
    print_status "Running tests in $service_name container..."
    
    # Test 1: Quick test
    print_status "Running quick test..."
    if docker-compose -f $compose_file exec -T audio-processor python quick_test.py; then
        print_success "Quick test passed"
    else
        print_error "Quick test failed"
        return 1
    fi
    
    # Test 2: Full test
    print_status "Running full test with results..."
    if docker-compose -f $compose_file exec -T audio-processor python test_with_full_results.py; then
        print_success "Full test passed"
    else
        print_error "Full test failed"
        return 1
    fi
    
    # Test 3: View results
    print_status "Viewing test results..."
    if docker-compose -f $compose_file exec -T audio-processor python view_results.py summary; then
        print_success "Results viewing passed"
    else
        print_error "Results viewing failed"
        return 1
    fi
    
    # Test 4: API health check
    print_status "Testing API health..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API health check passed"
    else
        print_error "API health check failed"
        return 1
    fi
    
    return 0
}

# Function to show service status
show_status() {
    local compose_file=$1
    
    print_status "Service status:"
    docker-compose -f $compose_file ps
    
    print_status "Service URLs:"
    echo "  - API: http://localhost:8000/docs"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Flower: http://localhost:5555"
    echo "  - MinIO: http://localhost:9001 (minioadmin/minioadmin)"
}

# Function to cleanup
cleanup() {
    local compose_file=$1
    print_status "Stopping services..."
    docker-compose -f $compose_file down
    print_success "Services stopped"
}

# Main function
main() {
    local mode=${1:-cpu}
    
    print_status "AudioProcessor Docker Testing Script"
    print_status "Mode: $mode"
    
    # Check Docker
    check_docker
    
    # Set compose file based on mode
    if [ "$mode" = "gpu" ]; then
        COMPOSE_FILE=$COMPOSE_GPU_FILE
        SERVICE_NAME="GPU"
        check_nvidia_docker
    else
        SERVICE_NAME="CPU"
    fi
    
    # Start services
    start_services $COMPOSE_FILE $SERVICE_NAME
    
    # Run tests
    if run_tests $COMPOSE_FILE $SERVICE_NAME; then
        print_success "All tests passed!"
        show_status $COMPOSE_FILE
    else
        print_error "Some tests failed"
        cleanup $COMPOSE_FILE
        exit 1
    fi
    
    print_status "Testing completed successfully!"
    print_status "Services are still running. Use 'docker-compose -f $COMPOSE_FILE down' to stop them."
}

# Handle script arguments
case "${1:-}" in
    "cpu")
        main "cpu"
        ;;
    "gpu")
        main "gpu"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [cpu|gpu|help]"
        echo ""
        echo "Options:"
        echo "  cpu   - Test CPU version (default)"
        echo "  gpu   - Test GPU version"
        echo "  help  - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0        # Test CPU version"
        echo "  $0 cpu    # Test CPU version"
        echo "  $0 gpu    # Test GPU version"
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
