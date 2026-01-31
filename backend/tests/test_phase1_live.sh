#!/bin/bash

# BRICK OS Backend - Phase 1 LIVE TEST
# Tests intelligent agent selection with real LLM-powered intent analysis

BASE_URL="http://localhost:8000"

echo "========================================="
echo "BRICK OS Backend - Phase 1 LIVE TEST"
echo "Testing with Real LLM Integration"
echo "========================================="
echo ""

# Live Test 1: Full Orchestrator with Agent Selection
echo "Live Test 1: Full Orchestrator - Simple Ball Design"
echo "----------------------------------------------------"
curl -X POST "${BASE_URL}/api/compile" \
  -H "Content-Type: application/json" \
  -d '{
    "user_intent": "design a simple rubber ball for a child",
    "project_id": "live-test-ball-001",
    "mode": "plan"
  }' | jq '.planning_doc, .environment'

echo ""
echo ""

# Live Test 2: Complex Autonomous System
echo "Live Test 2: Full Orchestrator - Autonomous Drone"
echo "--------------------------------------------------"
curl -X POST "${BASE_URL}/api/compile" \
  -H "Content-Type: application/json" \
  -d '{
    "user_intent": "design an autonomous quadcopter drone with GPS navigation and battery power for 30 minute flight time",
    "project_id": "live-test-drone-001",
    "mode": "plan"
  }' | jq '.planning_doc, .environment'

echo ""
echo ""

# Live Test 3: Medical Device
echo "Live Test 3: Full Orchestrator - Medical Implant"
echo "-------------------------------------------------"
curl -X POST "${BASE_URL}/api/compile" \
  -H "Content-Type: application/json" \
  -d '{
    "user_intent": "design a biocompatible hip implant that meets FDA medical device standards",
    "project_id": "live-test-medical-001",
    "mode": "plan"
  }' | jq '.planning_doc, .environment'

echo ""
echo ""
echo "========================================="
echo "Live tests complete!"
echo "Check the planning_doc output to see"
echo "real LLM-generated design plans."
echo "========================================="
