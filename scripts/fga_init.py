import os
import uuid
import asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv
from openfga_sdk import (
    ClientConfiguration,
    Metadata,
    RelationMetadata,
    RelationReference,
    TypeDefinition,
    WriteAuthorizationModelRequest,
    Userset,
    OpenFgaClient,
)
from openfga_sdk.client.models import ClientTuple
from openfga_sdk.credentials import CredentialConfiguration, Credentials

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize OpenFGA client configuration
def get_fga_client():
    fga_configuration = ClientConfiguration(
        api_url=os.getenv("FGA_API_URL") or "api.us1.fga.dev",
        store_id=os.getenv("FGA_STORE_ID"),
        credentials=Credentials(
            method="client_credentials",
            configuration=CredentialConfiguration(
                api_issuer=os.getenv("FGA_API_TOKEN_ISSUER") or "auth.fga.dev",
                api_audience=os.getenv("FGA_API_AUDIENCE") or "https://api.us1.fga.dev/",
                client_id=os.getenv("FGA_CLIENT_ID"),
                client_secret=os.getenv("FGA_CLIENT_SECRET"),
            ),
        ),
    )
    return OpenFgaClient(fga_configuration)

# Helper function to validate file type
def is_valid_file_type(file: UploadFile) -> bool:
    allowed_extensions = ["pdf", "txt"]
    file_extension = file.filename.split(".")[-1].lower()
    return file_extension in allowed_extensions

# Function to initialize FGA (model setup)
async def initialize_fga():
    fga_configuration = ClientConfiguration(
        api_url=os.getenv("FGA_API_URL") or "api.us1.fga.dev",
        store_id=os.getenv("FGA_STORE_ID"),
        credentials=Credentials(
            method="client_credentials",
            configuration=CredentialConfiguration(
                api_issuer=os.getenv("FGA_API_TOKEN_ISSUER") or "auth.fga.dev",
                api_audience=os.getenv("FGA_API_AUDIENCE") or "https://api.us1.fga.dev/",
                client_id=os.getenv("FGA_CLIENT_ID"),
                client_secret=os.getenv("FGA_CLIENT_SECRET"),
            ),
        ),
    )

    async with OpenFgaClient(fga_configuration) as fga_client:
        # Define 'user' type
        user_type = TypeDefinition(type="user")

        # Define relations for resume documents
        resume_relations = dict(
            owner=Userset(this=dict()),
            viewer=Userset(this=dict()),
        )

        resume_metadata = Metadata(
            relations=dict(
                owner=RelationMetadata(
                    directly_related_user_types=[
                        RelationReference(type="user"),
                    ]
                ),
                viewer=RelationMetadata(
                    directly_related_user_types=[
                        RelationReference(type="user"),
                        RelationReference(type="user", wildcard={}),
                    ]
                ),
            )
        )

        # Define 'resume' type
        resume_type = TypeDefinition(
            type="resume", relations=resume_relations, metadata=resume_metadata
        )

        # Authorization model request for FGA
        authorization_model_request = WriteAuthorizationModelRequest(
            schema_version="1.1",
            type_definitions=[user_type, resume_type],
            conditions=dict(),
        )

        model = await fga_client.write_authorization_model(authorization_model_request)
        print("NEW MODEL ID:", model)

        # Configuring predefined tuples for public access (resume template) and user-specific access
        await fga_client.write_tuples(
            body=[
                ClientTuple(user="user:*", relation="viewer", object="resume:public-template"),  # Public template
            ]
        )


# Endpoint for uploading resume
@app.post("/upload_resume/{user_id}")
async def upload_resume(user_id: str, file: UploadFile = File(...)):
    # Validate the file type (PDF, DOCX, TXT only)
    if not is_valid_file_type(file):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and TXT are allowed.")

    # Generate a unique identifier for the resume
    user_id= user_id.strip(" ")
    resume_id = f"{user_id}-{uuid.uuid4()}"

    # Save the uploaded file locally
    file_location = f"uploads/{resume_id}.{file.filename.split('.')[-1]}"
    try:
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving the file: {str(e)}")

    # Write the relationship in OpenFGA using write_tuples (only the owner can access the resume)
    try:
        async with get_fga_client() as fga_client:
            await fga_client.write_tuples(
                body=[
                    ClientTuple(user=f"user:{user_id}", relation="owner", object=f"resume:{resume_id}"),
                    ClientTuple(user=f"user:{user_id}", relation="viewer", object=f"resume:{resume_id}"),  # User can view their own resume
                ]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing FGA tuples: {str(e)}")

    return {"message": f"Resume uploaded successfully and linked to user {user_id}", "resume_id": resume_id}


# Run FastAPI app
if __name__ == "__main__":
    # Initialize FGA schema and tuples if not already initialized
    asyncio.run(initialize_fga())
    
    # Start the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
