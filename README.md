# AI_Projects

Repository for projects using AI models for various tasks locally.

This repository contains several projects leveraging AI models for audio, video, and language processing. The projects are organized into the following categories:

-   **Core Backend:** A FastAPI project (transitioning from Django) providing a backend with Supabase integration for remote generation and processing tasks.
-   **Audio Projects:** Projects focused on audio processing tasks, such as denoising and dereverberation.
-   **LLM Projects:** Projects utilizing Large Language Models (LLMs) for various tasks, such as RAG (Retrieval-Augmented Generation) and AI specialists.
-   **Visual Projects:** Projects focused on video and image processing tasks, such as greenscreening, denoising, and super-resolution.
-   **MCP Projects:** Model Context Protocol servers for enhancing AI capabilities, particularly with Claude Desktop integration.

## Current State

-   Topics generator for freestyle app (RAG + Ollama Polish Bielik model)
-   Video greenscreening model
-   MCP servers integration with Claude Desktop
-   Backend transition from Django DRF to FastAPI + Supabase

## Project Structure

The repository is organized into the following main directories:

-   **Core Backend:** Contains the FastAPI project (transitioning from Django) for backend services.
-   **Audio Projects:** Includes projects related to audio processing.
-   **LLM Projects:** Contains projects focused on Large Language Models.
-   **Visual Projects:** Includes projects for video and image processing.
-   **MCP Projects:** Contains Model Context Protocol servers for enhanced AI capabilities.
-   **memory-bank:** Stores project documentation and context.
-   **OpenManus:** Integration for orchestration with local models and external APIs.

## Technologies Used

-   **Backend:**
    -   FastAPI (transitioning from Django)
    -   Supabase
    -   Docker
    
-   **AI Models & Deployment:**
    -   Ollama
    -   PyTorch
    -   open-webui
    
-   **External APIs:**
    -   OpenRouter API
    -   Google Gemini API
    
-   **Integration & Workflow:**
    -   n8n
    -   Model Context Protocol (MCP)
    -   Claude Desktop

## Project Goals

The main goal of this project is to create AI-powered solutions for various needs and develop businesses that people are willing to invest in. This includes:

-   Developing AI specialists with low hardware requirements.
-   Enhancing long context memory agents.
-   Optimizing processes for businesses.
-   Creating new services such as AI Photographers.
-   Supporting artistic content creation.

## Current Focus

-   Agentic RAG development, LLM, Ollama pipelines.
-   MCP servers integration and configuration.
-   Backend transition to FastAPI + Supabase.
-   Audio dereverberation.
-   Video Greenscreening.

## Future Plans

-   Complete the FastAPI + Supabase backend implementation.
-   Expand MCP servers functionality and integration.
-   Implement OpenManus orchestration for local models and external APIs.
-   Develop models for Reaper processing (denoising, dereverberation).
-   Create comprehensive Docker deployment system.

## System Architecture

-   Python with one myenv
-   Docker containers
-   Supabase for database and authentication
-   MCP servers for extended AI capabilities
-   Cloned/Forked repositories