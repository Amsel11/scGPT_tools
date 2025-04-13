import torch
import torch.nn as nn
import time
import logging
import warnings
import numpy as np

def train_continual(
    model, 
    loader, 
    optimizer, 
    scheduler, 
    scaler, 
    device, 
    config, 
    vocab, 
    epoch, 
    replay_loaders=None,
    replay_weight=0.3,
    log_interval=100,
    logger=None
):
    """
    Train the model for one epoch with basic continual learning support.
    """
    # Setup for tracking metrics
    metrics = {
        "loss": 0.0,
        "mse": 0.0,
        "cls": 0.0,
        "mvc": 0.0,
        "replay_loss": 0.0,
    }
    
    # Set model to training mode
    model.train()
    total_batches = 0
    start_time = time.time()
    pad_token = config.pad_token
    mask_value = config.mask_value
    
    # Enable features based on configuration
    MLM = config.MLM
    CLS = config.CLS
    MVC = config.MVC
    
    # Define loss functions
    criterion = masked_mse_loss  # Make sure this is imported or defined
    criterion_cls = nn.CrossEntropyLoss()
    
    num_batches = len(loader)
    if logger:
        logger.info(f"Training on {num_batches} batches for epoch {epoch}")
    
    # Main training loop
    for batch, batch_data in enumerate(loader):
        # Move data to device
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        
        # Calculate padding mask
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if getattr(config, 'INPUT_BATCH_LABELS', False) else None,
                CLS=CLS,
                MVC=MVC,
            )

            # Identify masked positions for prediction
            masked_positions = input_values.eq(mask_value)
            
            # Initialize loss
            loss = 0.0
            
            # Calculate various losses based on enabled features
            if MLM:
                loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_mse
                metrics["mse"] += loss_mse.item()
                
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics["cls"] += loss_cls.item()
                
            if MVC:
                loss_mvc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_mvc
                metrics["mvc"] += loss_mvc.item()
                
            # Add replay mechanism for continual learning
            replay_loss = 0.0
            if replay_loaders and batch % 5 == 0:  # Process replay every 5 batches
                for replay_loader in replay_loaders.values():
                    try:
                        replay_batch = next(iter(replay_loader))
                    except StopIteration:
                        # Reset the iterator if we've gone through it
                        replay_loader = iter(replay_loader)
                        replay_batch = next(iter(replay_loader))
                        
                    # Process replay data
                    replay_gene_ids = replay_batch["gene_ids"].to(device)
                    replay_values = replay_batch["values"].to(device)
                    replay_target_values = replay_batch["target_values"].to(device)
                    replay_celltype_labels = replay_batch["celltype_labels"].to(device)
                    
                    replay_padding_mask = replay_gene_ids.eq(vocab[pad_token])
                    replay_masked_positions = replay_values.eq(mask_value)
                    
                    # Forward pass with replay data
                    replay_output_dict = model(
                        replay_gene_ids,
                        replay_values,
                        src_key_padding_mask=replay_padding_mask,
                        batch_labels=None,
                        CLS=CLS,
                        MVC=MVC,
                    )
                    
                    # Calculate replay losses (focusing on main objectives)
                    replay_batch_loss = 0.0
                    
                    if MLM:
                        replay_mse = criterion(
                            replay_output_dict["mlm_output"], 
                            replay_target_values, 
                            replay_masked_positions
                        )
                        replay_batch_loss += replay_mse
                        
                    if CLS:
                        replay_cls = criterion_cls(
                            replay_output_dict["cls_output"], 
                            replay_celltype_labels
                        )
                        replay_batch_loss += replay_cls
                    
                    if MVC:
                        replay_mvc = criterion(
                            replay_output_dict["mvc_output"], 
                            replay_target_values, 
                            replay_masked_positions
                        )
                        replay_batch_loss += replay_mvc
                        
                    replay_loss += replay_batch_loss
                
                # Scale and add the replay loss to the total loss
                if replay_loss > 0:
                    replay_loss = replay_loss * replay_weight
                    loss = loss + replay_loss
                    metrics["replay_loss"] += replay_loss.item()
        
        # Backward pass and optimization
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            getattr(config, 'max_grad_norm', 1.0),
            error_if_nonfinite=False if scaler.is_enabled() else True,
        )
        
        # Update model parameters
        scaler.step(optimizer)
        scaler.update()
        
        # Update total metrics
        metrics["loss"] += loss.item()
        total_batches += 1
        
        # Print progress
        if batch % log_interval == 0 and batch > 0 and logger:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            
            # Calculate current metrics
            cur_metrics = {k: v / log_interval for k, v in metrics.items() if v != 0}
            
            # Build log message
            log_msg = (
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_metrics.get('loss', 0):5.2f} | "
            )
            
            # Add optional components to log message
            if MLM:
                log_msg += f"mse {cur_metrics.get('mse', 0):5.2f} | "
            if CLS:
                log_msg += f"cls {cur_metrics.get('cls', 0):5.2f} | "
            if MVC:
                log_msg += f"mvc {cur_metrics.get('mvc', 0):5.2f} | "
            if "replay_loss" in cur_metrics:
                log_msg += f"replay {cur_metrics.get('replay_loss', 0):5.2f} | "
                
            logger.info(log_msg)
            
            # Reset metrics
            for k in metrics:
                metrics[k] = 0.0
            start_time = time.time()
    
    # Return average metrics
    return {k: v / total_batches for k, v in metrics.items() if v != 0}

def evaluate_continual(
    model, 
    loader, 
    device, 
    config, 
    vocab, 
    dataset_name=None,
    logger=None
):
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Define loss function
    criterion_cls = nn.CrossEntropyLoss()
    pad_token = config.pad_token
    
    if logger:
        logger.info(f"Evaluating model on {len(loader)} batches" + 
                   (f" for dataset {dataset_name}" if dataset_name else ""))
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            # Forward pass
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if getattr(config, 'INPUT_BATCH_LABELS', False) else None,
                CLS=True,  # We need classification for evaluation
                MVC=False,
            )
            
            # Calculate loss
            output_values = output_dict["cls_output"]
            loss = criterion_cls(output_values, celltype_labels)
            
            # Calculate accuracy
            predictions = output_values.argmax(dim=1)
            correct = (predictions == celltype_labels).sum().item()
            
            # Update totals
            total_loss += loss.item() * len(input_gene_ids)
            total_correct += correct
            total_samples += len(input_gene_ids)
    
    # Calculate final metrics
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    if logger:
        logger.info(f"Evaluation results: loss={avg_loss:.5f}, accuracy={accuracy:.5f}")

    return avg_loss, 1.0 - accuracy  # Return loss and error rate


def store_replay_data(loader, num_samples=1000, logger=None):
    """
    Store data for replay in continual learning.
    """
    if logger:
        logger.info(f"Storing replay data ({num_samples} samples)")
    
    replay_data = {
        "gene_ids": [],
        "values": [],
        "target_values": [],
        "batch_labels": [],
        "celltype_labels": []
    }
    
    collected_samples = 0
    
    for batch_data in loader:
        batch_size = batch_data["gene_ids"].size(0)
        samples_to_add = min(batch_size, num_samples - collected_samples)
        
        if samples_to_add <= 0:
            break
            
        # Store the data
        for key in replay_data:
            replay_data[key].append(batch_data[key][:samples_to_add].cpu())
            
        collected_samples += samples_to_add
        
        if collected_samples >= num_samples:
            break
    
    # Concatenate the data
    for key in replay_data:
        if replay_data[key]:
            replay_data[key] = torch.cat(replay_data[key], dim=0)
        else:
            replay_data[key] = torch.tensor([])
    
    if logger:
        logger.info(f"Stored {collected_samples} samples for replay")
        
    return replay_data