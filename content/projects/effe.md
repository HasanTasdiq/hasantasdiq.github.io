+++
title = "Efficient Multilingual Feature Extraction for Edge NLP"
date = 2025-09-25
tags = ["LLM","Jetson","EdgeAI","NLP"]
draft = false
+++

# LLM-Powered Feature Extraction for Edge

_Paper under review_

![Pareto Frontier](/projects/effe/eval_effe.png)



This research introduces a novel approach for deploying multilingual NLP on edge devices by rethinking the role of large language models. Instead of running full LLM inference, the framework leverages them as static feature repositories, bypassing the computationally expensive transformer stack.

The method enables a tunable efficiency hierarchy, from ultra-efficient static embeddings to enhanced hybrid features via lightweight distillation. By using multilingual LLMs as a unified source, it eliminates the complexity of managing language-specific pipelines, providing a scalable path to on-device intelligence without the traditional overhead.