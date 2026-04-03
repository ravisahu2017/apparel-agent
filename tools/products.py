import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.exc import SQLAlchemyError

from langchain.tools import tool

from .conn import Base, SessionLocal


class Product(Base):
    __tablename__ = "products"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attributes = Column(JSONB, nullable=True)
    listing = Column(JSONB, nullable=True)
    status = Column(String, default="draft")
    created_at = Column(DateTime, default=datetime.utcnow())
    updated_at = Column(DateTime, default=datetime.utcnow(), onupdate=datetime.utcnow())


@tool("create_product", description="Creates a new product")
def create_product(
    attributes: Optional[Dict[str, Any]] = None,
    listing: Optional[Dict[str, Any]] = None,
    status: str = "draft",
) -> Product:
    return _create_product(attributes, listing, status)


@tool("get_product", description="Fetches product details by ID")
def get_product(product_id: uuid.UUID) -> Optional[Product]:
    return _get_product(product_id)


@tool("update_product", description="Updates product details")
def update_product(
    product_id: uuid.UUID,
    attributes: Optional[Dict[str, Any]] = None,
    listing: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
) -> Optional[Product]:
    return _update_product(product_id, attributes, listing, status)


# ==============================
# CRUD Operations
# ==============================


def _create_product(
    attributes: Optional[Dict[str, Any]] = None,
    listing: Optional[Dict[str, Any]] = None,
    status: str = "draft",
) -> Product:
    session = SessionLocal()
    try:
        product = Product(
            attributes=attributes,
            listing=listing,
            status=status,
        )
        session.add(product)
        session.commit()
        session.refresh(product)
        return product
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def _get_product(product_id: uuid.UUID) -> Optional[Product]:
    """Internal function for getting product (can be called directly)"""
    session = SessionLocal()
    try:
        return session.query(Product).filter(Product.id == product_id).first()
    finally:
        session.close()


def _update_product(
    product_id: uuid.UUID,
    attributes: Optional[Dict[str, Any]] = None,
    listing: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
) -> Optional[Product]:
    """Internal function for updating product (can be called directly)"""
    print(
        f"Updating product {product_id} with attributes: {attributes}, listing: {listing}, status: {status}"
    )
    session = SessionLocal()
    try:
        product = session.query(Product).filter(Product.id == product_id).first()
        if not product:
            return None

        if attributes is not None:
            product.attributes = attributes

        if listing is not None:
            product.listing = listing

        if status is not None:
            product.status = status

        product.updated_at = datetime.utcnow()

        session.commit()
        session.refresh(product)
        return product

    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def _delete_product(product_id: uuid.UUID) -> bool:
    session = SessionLocal()
    try:
        product = session.query(Product).filter(Product.id == product_id).first()
        if not product:
            return False

        session.delete(product)
        session.commit()
        return True
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()
