![OpenAI](/logo.png)

# File Uploads FAQ

File Uploads FAQ

## About file uploads

ChatGPT can work with documents, spreadsheets, presentations, and other supported files you upload in a conversation.

## Availability

File uploads are available on Free and paid ChatGPT plans, subject to plan-specific usage limits and account settings, on the web and in supported mobile apps. File uploads are also available through the API for supported API features, with separate API requirements, supported purposes, and limits.

## How does the new file uploads capability work?

The file uploads capability was created to support the following tasks:  
  
  
**Synthesis:** Combining or analyzing information from files and documents to create something new, for example:

Upload a spreadsheet, for example a CSV, with a mix of qualitative and quantitative information, and ask ChatGPT to help you understand and visualize the data.

Compare and contrast two documents.

Analyze sentiment or tone in a document.

Analyze a spreadsheet.

Apply a framework or rubric from one document to the contents of another.

**Transformation:** Reshaping information from documents without changing its essence, for example:

Upload a complicated research paper and ask ChatGPT to provide a simple summary.

Upload a PowerPoint presentation and ask ChatGPT for feedback on the content.

Summarize a document in simple terms.

Rewrite a short document in a particular style.

Turn a presentation into a document.

**Extraction:** Pulling specific information out of a document, for example:

Upload a PDF and have ChatGPT find any references to a certain topic.

Pull out relevant quotes from a document.

Search for any mention of a particular topic from a document or spreadsheet.

Extract metadata (author, creation date, etc.) from a document.

Count the number of rows in a spreadsheet that contain a certain attribute.

Extract specific sections of a document (e.g., all headings or all bullet-point lists).

## What types of files are supported?

All common file extensions for text files, spreadsheets, presentations, and documents.

## How many files can I upload at once per GPT?

Up to **10** files per GPT for the lifetime of that GPT. Keep in mind there are file size restrictions and usage caps per user/org.

## What are the file upload size restrictions?

All files uploaded to a GPT or a ChatGPT conversation have a hard limit of 512MB per file.

All text and document files uploaded to a GPT or to a ChatGPT conversation are capped at 2M tokens per file. This limitation does not apply to spreadsheets.

For CSV files or spreadsheets, the file size cannot exceed approximately 50MB, depending on the size of each row.

For images, there's a limit of 20MB per image.

Additionally, there are usage caps:

Each end-user is capped at 25GB.

Each organization is capped at 100GB.

Note: An error will be displayed if a user/org cap has been hit.

Users can upload up to 80 files every 3 hours. Free users are limited to 3 file uploads per day. Note that we may lower these limits during peak hours.  
  
Troubleshooting “upload limit reached”  
  
If you see an “upload limit reached” message despite few or no recent uploads, try the steps below:

Confirm you’re signed into the correct account and subscription plan.

Remember there are two kinds of limits: a rolling upload rate (e.g., up to 80 files per 3 hours) and shared storage caps (25 GB per user / 100 GB per org) that apply across chats, Projects, and custom GPT knowledge.

Failed upload attempts can sometimes count toward the upload-rate cap.

Check status.openai.com for incidents that may affect uploads.

To escalate, include your account email, a screenshot, the timestamp and timezone, the platform/app/browser you used, and the request ID (if available).

Currently, you can check your Library storage usage in ChatGPT under Settings > Storage. ChatGPT does not currently show how much of your rolling upload-rate quota has been used or how much remains.  
  
Additionally, to help ensure a smooth experience for all users, we've also updated the file upload limits for *Projects* based on subscription plan.

**Plus**: Up to 20 files per project

**Pro, Team, Education, Business**: Up to 40 files per project

## How do I delete files I upload?

Files uploaded to Advanced Data Analysis are deleted within a duration that varies based on your plan. If you are encountering your file usage cap, you can also delete files from recent chats or from any GPTs that you built, as these share caps.

## How are files vs chats retained?

Chats

Chats are saved in your account until you delete them. To learn more about data controls, see: [Data Controls FAQ](https://help.openai.com/articles/7730893).

Once you delete a chat or delete your account, chats are deleted from our systems within 30 days, unless they have previously been de-identified and disassociated from your account, or we have to keep them for security or legal reasons.

Files

Files uploaded to ChatGPT are saved in your account up to the retention period of the corresponding chat. Files uploaded as knowledge to a custom GPT are retained until you delete the custom GPT.

Once you delete a chat containing a file, your account, or a custom GPT, the associated file is deleted from our systems within 30 days, unless it has previously been de-identified and disassociated from your account, or we have to keep it for security or legal reasons.

Files processed via ADA / Document Analysis, and when chatting with a custom GPT (not uploaded as knowledge in GPT config): Retained for a duration that varies based on your plan.

For more information about file and chat retention, see: [Chat and File Retention Policies in ChatGPT](https://help.openai.com/articles/8983778).

## Are you able to handle images embedded in docs/presentations/PDF?

The answer depends on your plan and the file type you're working with.  
  
ChatGPT Enterprise supports Visual Retrieval for PDF files. To learn more about PDF visual retrieval, see: [Visual Retrieval with PDFs FAQ](https://help.openai.com/articles/10416312).  
  
All other plans and document files only support text-based retrieval. This means that ChatGPT will extract digital text from the file and discard any images.

## Will OpenAI use files uploaded to train its models?

The answer depends on the service you are using. For more information about data usage for consumer services, see: [Data Usage for Consumer Services FAQ](https://help.openai.com/articles/7039943). Content may include files that are uploaded. To learn more about how content may be used to improve model performance and the choices you have, see: [How your data is used to improve model performance](https://help.openai.com/articles/5722486).  
  
Please note that we do not use content submitted by customers to our business offerings such as our API and ChatGPT Enterprise to improve model performance.  
  
Please see our [Enterprise Privacy page](https://openai.com/enterprise-privacy) for information on how we use business data.

## Was this article helpful?

![OpenAI logo](/favicon.png)
