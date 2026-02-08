"""
BRICK OS - Service Layer

Centralized services for database, external APIs, and business logic.
All agents should use these services instead of hardcoded values.

Usage:
    from backend.services import supabase, pricing_service, standards_service
    
    # Get thresholds
    thresholds = await supabase.get_critic_thresholds("ControlCritic", "drone_large")
    
    # Get material price
    price = await pricing_service.get_material_price("Aluminum 6061")
    
    # Get standards
    fit = await standards_service.get_iso_fit("H7/g6")
"""

from .supabase_service import SupabaseService, supabase
from .pricing_service import PricingService, pricing_service
from .standards_service import StandardsService, standards_service
from .component_catalog_service import ComponentCatalogService, component_catalog
from .asset_sourcing_service import AssetSourcingService, asset_sourcing
from .currency_service import CurrencyService, currency_service

__all__ = [
    # Classes
    "SupabaseService",
    "PricingService",
    "StandardsService",
    "ComponentCatalogService",
    "AssetSourcingService",
    "CurrencyService",
    # Singleton instances
    "supabase",
    "pricing_service",
    "standards_service",
    "component_catalog",
    "asset_sourcing",
    "currency_service",
]
