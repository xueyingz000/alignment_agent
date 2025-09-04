# 导入代理类
from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG

# 初始化代理
agent = IFCSemanticAgentNoKG()

# 处理查询
response = agent.process_query("How to map IfcWall to regulation terminology?")#!/usr/bin/env python3
"""Test script for IFC Semantic Agent without Knowledge Graph."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.ifc_semantic_agent_no_kg import IFCSemanticAgentNoKG
from utils import get_logger

def test_agent_initialization():
    """Test agent initialization without knowledge graph."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing IFC Semantic Agent initialization (No KG version)...")
        
        # Initialize agent
        agent = IFCSemanticAgentNoKG()
        
        logger.info("✓ Agent initialized successfully")
        
        # Check agent state
        state = agent.get_agent_state()
        logger.info(f"Agent state: {state}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Agent initialization failed: {e}")
        return False

def test_simple_query():
    """Test a simple query processing."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing simple query processing...")
        
        # Initialize agent
        agent = IFCSemanticAgentNoKG()
        
        # Test query
        query = "What is the relationship between IfcWall and building regulations?"
        
        logger.info(f"Processing query: {query}")
        
        # Process query
        response = agent.process_query(query)
        
        logger.info(f"✓ Query processed successfully")
        logger.info(f"Final answer: {response.final_answer}")
        logger.info(f"Confidence: {response.confidence_score:.2f}")
        logger.info(f"Steps taken: {response.total_steps}")
        logger.info(f"Execution time: {response.execution_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Query processing failed: {e}")
        return False

def test_with_sample_data():
    """Test with sample IFC and regulatory data."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing with sample data...")
        
        # Initialize agent
        agent = IFCSemanticAgentNoKG()
        
        # Sample IFC data
        sample_ifc_data = {
            'entities': {
                'wall_001': {
                    'type': 'IfcWall',
                    'attributes': {'Name': 'ExteriorWall', 'Description': 'Load bearing wall'},
                    'properties': {'Width': 200, 'Height': 3000, 'Material': 'Concrete'}
                },
                'slab_001': {
                    'type': 'IfcSlab',
                    'attributes': {'Name': 'FloorSlab', 'Description': 'Structural floor slab'},
                    'properties': {'Thickness': 150, 'Material': 'Reinforced Concrete'}
                }
            },
            'relationships': [
                {'source': 'wall_001', 'target': 'slab_001', 'type': 'supports'}
            ]
        }
        
        # Sample regulatory text
        sample_regulatory_text = """
        Building Code Section 3.2: Wall Requirements
        All load-bearing walls shall have a minimum thickness of 150mm.
        Exterior walls must provide adequate thermal insulation.
        Floor slabs shall be designed to support the required live loads.
        """
        
        # Test query
        query = "How do the IFC wall and slab entities align with building code requirements?"
        
        logger.info(f"Processing query with sample data: {query}")
        
        # Process query with data
        response = agent.process_query(
            query=query,
            ifc_data=sample_ifc_data,
            regulatory_text=sample_regulatory_text
        )
        
        logger.info(f"✓ Query with sample data processed successfully")
        logger.info(f"Final answer: {response.final_answer}")
        logger.info(f"Confidence: {response.confidence_score:.2f}")
        logger.info(f"Steps taken: {response.total_steps}")
        logger.info(f"Knowledge sources: {response.knowledge_sources}")
        logger.info(f"Semantic mappings: {len(response.semantic_mappings)}")
        
        # Show ReAct steps
        logger.info("\nReAct Steps:")
        for i, step in enumerate(response.react_steps):
            logger.info(f"Step {i+1}: {step.action.action_type.value} - {step.thought.reasoning_type}")
            logger.info(f"  Reasoning: {step.action.reasoning}")
            logger.info(f"  Success: {step.success}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Sample data processing failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function."""
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("Testing IFC Semantic Agent (No Knowledge Graph Version)")
    logger.info("=" * 60)
    
    tests = [
        ("Agent Initialization", test_agent_initialization),
        ("Simple Query", test_simple_query),
        ("Sample Data Processing", test_with_sample_data)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
        
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary:")
    logger.info(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! The No-KG agent is working correctly.")
        return True
    else:
        logger.error(f"❌ {len(results) - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)