import pymupdf as fitz
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List, Dict, Any
import io

class MultimodalDocumentParser:
    def __init__(self, model_id: str = "microsoft/Florence-2-base"):
        self.device = "cpu"
        self.torch_dtype = torch.float32

        print(f"Loading Florence-2 on {self.device}...")

        import torch.nn as nn
        if not hasattr(nn.Module, "_supports_sdpa"):
            nn.Module._supports_sdpa = False

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device).eval()

        if not hasattr(self.model, "_supports_sdpa"):
            self.model._supports_sdpa = False

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Model loaded successfully.")

    def parse_pdf(self, pdf_path: str, max_pages: int = 2) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        all_chunks = []
        pages_to_process = min(len(doc), max_pages)

        for page_idx in range(pages_to_process):
            page_num = page_idx + 1
            print(f"Processing Page {page_num}/{pages_to_process}...")
            page = doc[page_idx]
            page_width, page_height = page.rect.width, page.rect.height

            # 1. Native digital PDF block extraction (Fast & precise)
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            page_chunks = []

            for b in blocks:
                text = b[4].strip()
                if text and len(text) > 10:  # ignore blank or tiny artifacts
                    page_chunks.append({
                        "page": page_num,
                        "content": text.replace("\n", " "),
                        "bbox": {
                            "x1": round(b[0] / page_width, 4),
                            "y1": round(b[1] / page_height, 4),
                            "x2": round(b[2] / page_width, 4),
                            "y2": round(b[3] / page_height, 4)
                        }
                    })

            # 2. If the page had no digital text (e.g., scanned PDF), fall back to Florence OCR with regions
            if not page_chunks:
                print(f"  [Fallback to Florence-2 OCR on Page {page_num}]")
                pix = page.get_pixmap(dpi=120)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                page_chunks = self._florence_ocr_region(img, page_num)

            all_chunks.extend(page_chunks)

        doc.close()
        return all_chunks

    def _florence_ocr_region(self, image: Image.Image, page_num: int) -> List[Dict[str, Any]]:
        square_img = image.resize((768, 768), Image.Resampling.BICUBIC)
        task_prompt = "<OCR_WITH_REGION>"
        inputs = self.processor(text=task_prompt, images=square_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                use_cache=False
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_output = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(768, 768)
        )

        elements = []
        raw = parsed_output.get("<OCR_WITH_REGION>", {})
        bboxes = raw.get("quad_boxes", []) or raw.get("bboxes", [])
        labels = raw.get("labels", [])

        for bbox, label in zip(bboxes, labels):
            if label.strip():
                # Handle both 4-point boxes and bounding rects
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[4], bbox[5]

                elements.append({
                    "page": page_num,
                    "content": label.strip(),
                    "bbox": {
                        "x1": round(x1 / 768.0, 4),
                        "y1": round(y1 / 768.0, 4),
                        "x2": round(x2 / 768.0, 4),
                        "y2": round(y2 / 768.0, 4)
                    }
                })
        return elements