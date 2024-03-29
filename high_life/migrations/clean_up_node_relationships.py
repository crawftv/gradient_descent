from llama_index.core.vector_stores.utils import metadata_dict_to_node

from high_life.settings import chroma_collection


def remove_invalid_relationships():
    # Get all the node IDs from the Chroma collection
    node_ids = chroma_collection.get()["ids"]
    for node_id in node_ids:
        # Retrieve the node from the Chroma collection
        node = chroma_collection.get(ids=node_id)
        node = metadata_dict_to_node(node["metadatas"][0])

        # Iterate over each relationship type and related node IDs
        for relationship_type, related_node_ids in node.relationships.items():
            # Check if the related nodes exist in the Chroma collection
            existing_related_node_ids = chroma_collection.get(ids=[related_node_ids.node_id])['ids']
            # Remove the relationship if the related node doesn't exist
            invalid_node_ids = set(related_node_ids) - set(existing_related_node_ids)
            if invalid_node_ids:
                node.relationships[relationship_type] = list(set(related_node_ids) - invalid_node_ids)

                # Update the node in the Chroma collection
                chroma_collection.update(ids=node_id, documents=node.text, metadatas=node.metadata)


if __name__ == "__main__":
    remove_invalid_relationships()
