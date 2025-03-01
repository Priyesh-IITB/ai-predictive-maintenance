#!/bin/bash
# Test low risk scenario
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"vibration":0.5, "temperature":90.5, "pressure":60.2}'

# Test high risk scenario  
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"vibration":1.9, "temperature":135.7, "pressure":88.3}'
