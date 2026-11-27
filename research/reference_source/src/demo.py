"""Interactive research demo with scene routing, reviews, and Grad-CAM."""
import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from threading import Lock

import gradio as gr
import numpy as np
import torch

from common import CLASSES, create_model, transform
from gradcam import GradCAM, heatmap_overlay
from scene_routing import (EXPERIMENT, FROZEN_CHECKPOINTS, MONITOR_PATH, ReviewStore,
                           displayed_size, route_scene)


UNIFIED_CHECKPOINT = (
    EXPERIMENT / 'unified_ac_s42' / 'best.pt',
    '9755d64cc689ffa2f8bd0d8b0c99ec935fbcd6f5112f474be64d10b1261c3400',
)


class DemoClassifier:
    def __init__(self, checkpoint, expected_sha256=None):
        with Path(checkpoint).open('rb') as stream:
            if expected_sha256 is not None:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != expected_sha256:
                    raise ValueError(f'Frozen checkpoint SHA-256 mismatch: {checkpoint}')
                stream.seek(0)
            weights = torch.load(stream, map_location='cpu', weights_only=True)
        if weights['model_name'] != 'resnet18':
            raise ValueError('Grad-CAM demo requires a ResNet18 checkpoint')
        if 'classes' in weights and tuple(weights['classes']) != CLASSES:
            raise ValueError('Checkpoint class ordering does not match current code')
        model = create_model(weights['model_name'], False)
        model.load_state_dict(weights['model_state'])
        self.cam = GradCAM(model.eval())
        self.preprocessing = transform()
        # Grad-CAM stores activations and gradients on the shared classifier.
        self.lock = Lock()

    def classify(self, image):
        if image is None:
            return {}, None
        rgb = image.convert('RGB')
        tensor = self.preprocessing(rgb).unsqueeze(0)
        with self.lock:
            heat, _, probability = self.cam.map(tensor)
        return ({label: float(probability[index]) for index, label in enumerate(CLASSES)},
                np.asarray(heatmap_overlay(rgb, heat)))


def classify_capture(image, scene, classifiers, reviews):
    if image is None:
        raise ValueError('Upload an image before classifying.')
    route = route_scene(scene)
    if 'AC' in classifiers:
        route = replace(route, variant='AC')
    probabilities, overlay = classifiers[route.variant].classify(image)
    predicted_label = max(probabilities, key=probabilities.get)
    prediction_id = reviews.save_prediction(route, predicted_label, displayed_size(image))
    metadata = {
        'prediction_id': prediction_id,
        'scene': route.scene,
        'scene_missing': route.scene_missing,
        'model_variant': route.variant,
        'checkpoint_seed': 42,
        'manual_review_required': route.scene_missing,
    }
    return probabilities, overlay, metadata


def build_page(classifier=None, *, classifiers=None, reviews=None):
    routed = classifier is None

    def classify(image, scene):
        try:
            probabilities, overlay, metadata = classify_capture(image, scene, classifiers, reviews)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        return probabilities, overlay, metadata, metadata['prediction_id'], None, ''

    def record_review(prediction_id, label):
        try:
            reviews.save_review(prediction_id, label)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        return 'Review saved.', reviews.field_metrics()

    with gr.Blocks(title='RobustLeafLab: Field Generalization') as page:
        gr.Markdown('# RobustLeafLab\nResearch demo for tomato-leaf photos.')
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type='pil', label='Tomato leaf photograph')
                if routed:
                    capture_scene = gr.Dropdown(
                        [('Field', 'field'), ('Greenhouse', 'greenhouse')], value=None,
                        label='Capture scene', info='Leave blank if the capture scene is unknown.')
                classify_button = gr.Button('Classify and explain')
            with gr.Column():
                prediction = gr.Label(label='Three-class prediction', num_top_classes=3)
                overlay = gr.Image(label='Grad-CAM: influential image areas')
                if routed:
                    route_metadata = gr.JSON(label='Prediction and routing metadata')
        if routed:
            gr.Markdown('### Human review\nFor a confirmed label, use the ID associated with the reviewed image.')
            prediction_id = gr.Textbox(label='Prediction ID')
            confirmed_label = gr.Dropdown(list(CLASSES), value=None, label='Confirmed label')
            review_button = gr.Button('Save review')
            review_status = gr.Textbox(label='Review status', interactive=False)
            metrics = gr.JSON(label='Field metrics from confirmed reviews')
            refresh = gr.Button('Refresh field metrics')
            classify_button.click(classify, [input_img, capture_scene],
                                  [prediction, overlay, route_metadata, prediction_id,
                                   confirmed_label, review_status], api_name='classify')
            review_button.click(record_review, [prediction_id, confirmed_label],
                                [review_status, metrics], api_name='record_review')
            refresh.click(reviews.field_metrics, outputs=metrics, api_name='field_metrics')
            page.load(reviews.field_metrics, outputs=metrics, api_name=False)
        else:
            classify_button.click(classifier.classify, input_img, [prediction, overlay])
        gr.Markdown('Limited to healthy, early blight and late blight. '
                    'Grad-CAM indicates influential regions.')
    return page


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--checkpoint', help='Use a single checkpoint as in the original demo.')
    mode.add_argument('--unified-ac', action='store_true',
                      help='Use the frozen AC42 candidate for all capture scenes, with review records.')
    mode.add_argument('--plantdoc-audit', action='store_true',
                      help='Use frozen A for manual audit predictions outside field monitoring.')
    parser.add_argument('--monitor-db', type=Path)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    if args.checkpoint:
        page = build_page(DemoClassifier(args.checkpoint))
    elif args.plantdoc_audit:
        page = build_page(DemoClassifier(*FROZEN_CHECKPOINTS['A']))
    else:
        checkpoints = {'AC': UNIFIED_CHECKPOINT} if args.unified_ac else FROZEN_CHECKPOINTS
        classifiers = {name: DemoClassifier(*checkpoint) for name, checkpoint in checkpoints.items()}
        default_db = UNIFIED_CHECKPOINT[0].with_name('events.sqlite3') if args.unified_ac else MONITOR_PATH
        page = build_page(classifiers=classifiers, reviews=ReviewStore(args.monitor_db or default_db))
    page.launch(server_port=args.port)


if __name__ == '__main__':
    main()
