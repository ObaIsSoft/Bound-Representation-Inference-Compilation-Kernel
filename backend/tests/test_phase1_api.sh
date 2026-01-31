#!/bin/bash

# BRICK OS Backend - Phase 1 API Tests
# Tests for ISA Handshake Protocol and Agent Selection

BASE_URL="http://localhost:8000"

echo "========================================="
echo "BRICK OS Backend - Phase 1 API Tests"
echo "========================================="
echo ""

# Test 1: ISA Handshake - Compatible Version
echo "Test 1: ISA Handshake (Compatible Version)"
echo "-------------------------------------------"
curl -X POST "${BASE_URL}/api/handshake" \
  -H "Content-Type: application/json" \
  -d '{
    "client_version": "1.0.0",
    "client_id": "test-client-001",
    "requested_features": ["scoped_execution", "intelligent_agent_selection"]
  }' | jq '.'

echo ""
echo ""

# Test 2: Agent Selection - Simple Design (Ball)
echo "Test 2: Agent Selection - Simple Design (Ball)"
echo "------------------------------------------------"
curl -X POST "${BASE_URL}/api/agents/select" \
  -H "Content-Type: application/json" \
  -d '{
    "user_intent": "design a rubber ball",
    "environment": {"type": "GROUND"},
    "design_parameters": {"num_components": 1}
  }' | jq '.'

echo ""
echo ""

# Test 3: Agent Selection - Complex Design (Drone)
echo "Test 3: Agent Selection - Complex Design (Drone)"
echo "--------------------------------------------------"
curl -X POST "${BASE_URL}/api/agents/select" \
  -H "Content-Type: application/json" \
  -d '{
    "user_intent": "design an autonomous drone with battery power",
    "environment": {"type": "AERIAL"},
    "design_parameters": {"num_components": 15, "complexity": "complex"}
  }' | jq '.'

echo ""
echo ""

# Test 4: Agent Selection - Medical Device
echo "Test 4: Agent Selection - Medical Device"
echo "-----------------------------------------"
curl -X POST "${BASE_URL}/api/agents/select" \
  -H "Content-Type: application/json" \
  -d '{
    "user_intent": "design a medical implant that is FDA compliant",
    "environment": {"type": "GROUND"},
    "design_parameters": {"num_components": 3}
  }' | jq '.'

echo ""
echo ""

# Test 5: Schema Version
echo "Test 5: Schema Version"
echo "----------------------"
curl -X GET "${BASE_URL}/api/schema/version" | jq '.'

echo ""
echo ""

# Test 6: ISA Handshake - Incompatible Version
echo "Test 6: ISA Handshake (Incompatible Version)"
echo "---------------------------------------------"
curl -X POST "${BASE_URL}/api/handshake" \
  -H "Content-Type: application/json" \
  -d '{
    "client_version": "2.0.0",
    "client_id": "test-client-002"
  }' | jq '.'

echo ""
echo ""
echo "========================================="
echo "All tests complete!"
echo "========================================="
