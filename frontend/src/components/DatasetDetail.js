import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  Text,
  Button,
  Spinner,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Card,
  CardBody,
  Stack,
  StackDivider,
  Stat,
  StatLabel,
  StatNumber,
  StatGroup,
  Alert,
  AlertIcon,
  Flex,
  IconButton,
  Icon,
  useColorModeValue,
  SimpleGrid,
  Textarea,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  VStack,
  HStack,
  Code,
  List,
  ListItem,
  ListIcon,
  InputGroup,
  Input,
  InputLeftElement,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Checkbox,
} from '@chakra-ui/react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FiArrowLeft, FiDownload, FiEdit, FiSave, FiX, FiCheck, FiSearch, FiTrash2, FiBox, FiZap, FiFileText, FiMessageSquare, FiLink } from 'react-icons/fi';
import { getDatasetDetails, getDatasetExamples, updateExample, searchExamples } from '../services/api';

const DatasetDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [editingExample, setEditingExample] = useState(null);
  const [editContent, setEditContent] = useState(null);
  const [selectedExportFormat, setSelectedExportFormat] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [minQuality, setMinQuality] = useState(0);
  const [maxQuality, setMaxQuality] = useState(10);
  const [showQualityFilter, setShowQualityFilter] = useState(false);
  const [selectedExamples, setSelectedExamples] = useState([]);
  const [isSelectAll, setIsSelectAll] = useState(false);
  const toast = useToast();
  const queryClient = useQueryClient();
  const { isOpen: isExportOpen, onOpen: onExportOpen, onClose: onExportClose } = useDisclosure();

  // Auto-open export modal if export=true in URL
  useEffect(() => {
    if (searchParams.get('export') === 'true') {
      onExportOpen();
    }
  }, [searchParams, onExportOpen]);

  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch dataset details
  const {
    data: dataset,
    isLoading: isLoadingDetails,
    isError: isErrorDetails,
    error: detailsError
  } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => getDatasetDetails(id)
  });

  // Fetch dataset examples (or search results)
  const {
    data: examplesData,
    isLoading: isLoadingExamples,
    isError: isErrorExamples,
    error: examplesError
  } = useQuery({
    queryKey: ['dataset-examples', id, page, pageSize, searchQuery, minQuality, maxQuality],
    queryFn: () => {
      const params = { page, page_size: pageSize };

      // Add quality filters if not default
      if (minQuality > 0 || maxQuality < 10) {
        params.min_quality = minQuality;
        params.max_quality = maxQuality;
      }

      if (searchQuery && searchQuery.trim()) {
        setIsSearching(true);
        return searchExamples(id, searchQuery, params);
      }
      setIsSearching(false);
      return getDatasetExamples(id, params);
    },
    enabled: !!dataset
  });

  // Update example mutation
  const updateExampleMutation = useMutation({
    mutationFn: ({ exampleId, content }) => {
      return fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/datasets/${id}/examples/${exampleId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      }).then(response => {
        if (!response.ok) {
          return response.json().then(err => {
            throw new Error(err.detail || 'Failed to update example');
          });
        }
        return response.json();
      });
    },
    onSuccess: (data, variables) => {
      // Update the cache with the new data
      queryClient.setQueryData(['dataset-examples', id, page, pageSize], (oldData) => {
        if (!oldData) return oldData;

        return {
          ...oldData,
          examples: oldData.examples.map((example) =>
            example.id === variables.exampleId
              ? { ...example, content: variables.content }
              : example
          ),
        };
      });

      toast({
        title: 'Example updated',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      setEditingExample(null);
      setEditContent(null);
    },
    onError: (error) => {
      toast({
        title: 'Failed to update example',
        description: error.message || 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
  });

  const handlePreviousPage = () => {
    setPage(Math.max(page - 1, 1));
  };

  const handleNextPage = () => {
    if (examplesData?.total_pages && page < examplesData.total_pages) {
      setPage(page + 1);
    }
  };

  const downloadDataset = () => {
    window.open(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/datasets/${id}/download`);
  };

  const exportFormats = [
    {
      name: 'HuggingFace',
      key: 'huggingface',
      description: 'HuggingFace datasets format with metadata',
      icon: FiBox,
      useCases: ['Upload to HuggingFace Hub', 'Share datasets publicly', 'Use with transformers library'],
      features: ['dataset_info.json', 'README.md', 'Push to hub script'],
    },
    {
      name: 'OpenAI',
      key: 'openai',
      description: 'OpenAI fine-tuning format (JSONL)',
      icon: FiZap,
      useCases: ['Fine-tune GPT-3.5/GPT-4', 'OpenAI API training'],
      features: ['Chat format', 'Train/val split', 'Validation report', 'Cost estimation'],
    },
    {
      name: 'Alpaca',
      key: 'alpaca',
      description: 'Alpaca instruction format (JSON)',
      icon: FiFileText,
      useCases: ['Instruction tuning', 'Fine-tune LLaMA/Alpaca models'],
      features: ['Instruction-input-output structure', 'Compatible with FastChat'],
    },
    {
      name: 'ShareGPT',
      key: 'sharegpt',
      description: 'ShareGPT conversation format (JSON)',
      icon: FiMessageSquare,
      useCases: ['Chat model training', 'Vicuna/ShareGPT format'],
      features: ['Human-GPT conversation structure', 'Multi-turn dialogues'],
    },
    {
      name: 'LangChain',
      key: 'langchain',
      description: 'LangChain document format (JSONL)',
      icon: FiLink,
      useCases: ['RAG systems', 'Vector databases', 'LangChain applications'],
      features: ['Documents/chat/qa_pairs modes', 'Vector store ready', 'Loader scripts'],
    },
  ];

  const handleExport = (format) => {
    const url = `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/datasets/${id}/export`;
    const params = new URLSearchParams({
      format: format.key,
      output_filename: `${dataset?.name}_${format.key}`,
    });

    window.open(`${url}?${params.toString()}`);

    toast({
      title: `Exporting to ${format.name}`,
      description: 'Your download will start shortly',
      status: 'success',
      duration: 3000,
      isClosable: true,
    });

    onExportClose();
  };

  const handleSelectExample = (exampleId) => {
    setSelectedExamples(prev => {
      if (prev.includes(exampleId)) {
        return prev.filter(id => id !== exampleId);
      }
      return [...prev, exampleId];
    });
  };

  const handleSelectAll = () => {
    if (isSelectAll) {
      setSelectedExamples([]);
      setIsSelectAll(false);
    } else {
      const allIds = examplesData?.examples?.map(ex => ex.id) || [];
      setSelectedExamples(allIds);
      setIsSelectAll(true);
    }
  };

  const handleBulkDelete = async () => {
    if (!selectedExamples.length) return;

    if (!window.confirm(`Delete ${selectedExamples.length} selected examples? This cannot be undone.`)) {
      return;
    }

    try {
      // Call bulk delete API (to be implemented)
      await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/datasets/${id}/examples/bulk`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ example_ids: selectedExamples })
      });

      toast({
        title: 'Examples deleted',
        description: `${selectedExamples.length} examples removed`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      setSelectedExamples([]);
      setIsSelectAll(false);
      queryClient.invalidateQueries(['dataset-examples', id]);
    } catch (error) {
      toast({
        title: 'Delete failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleBulkExport = () => {
    if (!selectedExamples.length) return;

    const url = `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/datasets/${id}/examples/export`;
    const params = new URLSearchParams({
      example_ids: selectedExamples.join(','),
      format: 'jsonl'
    });

    window.open(`${url}?${params.toString()}`);

    toast({
      title: 'Exporting selected examples',
      description: `${selectedExamples.length} examples`,
      status: 'info',
      duration: 3000,
      isClosable: true,
    });
  };

  const handleEditExample = (example) => {
    setEditingExample(example);
    setEditContent(JSON.stringify(example.content, null, 2));
  };

  const handleSaveExample = async () => {
    try {
      const content = JSON.parse(editContent);
      updateExampleMutation.mutate({
        exampleId: editingExample.id,
        content: content,
      });
    } catch (error) {
      toast({
        title: 'Invalid JSON format',
        description: 'Please check your JSON syntax',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const formatExample = (example) => {
    if (!example) return null;

    // If this example is being edited, show edit form
    if (editingExample && editingExample.id === example.id) {
      return (
        <Box>
          <Textarea
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            minHeight="200px"
            fontFamily="monospace"
            mb={2}
          />
          <Flex justify="flex-end" gap={2}>
            <Button
              size="sm"
              leftIcon={<FiSave />}
              colorScheme="blue"
              onClick={handleSaveExample}
              isLoading={updateExampleMutation.isLoading}
            >
              Save
            </Button>
            <Button
              size="sm"
              leftIcon={<FiX />}
              onClick={() => {
                setEditingExample(null);
                setEditContent(null);
              }}
            >
              Cancel
            </Button>
          </Flex>
        </Box>
      );
    }

    // Extract content from example
    const content = example.content || example;

    if (dataset?.format === 'chat') {
      return (
        <Card p={4} variant="outline" borderWidth={1} borderRadius="md" boxShadow="sm">
          <Stack spacing={3} divider={<StackDivider />}>
            {content.messages?.map((message, idx) => (
              <Box key={idx} p={2} bg={message.role === 'user' ? 'gray.50' : 'white'} borderRadius="md">
                <Flex align="center" justify="space-between" mb={2}>
                  <Badge colorScheme={message.role === 'user' ? 'blue' : 'green'} fontSize="sm">
                    {message.role}
                  </Badge>
                  <IconButton
                    icon={<FiEdit />}
                    size="sm"
                    variant="ghost"
                    onClick={() => handleEditExample(example)}
                    aria-label="Edit example"
                  />
                </Flex>
                <Text whiteSpace="pre-wrap" fontSize="md">{message.content}</Text>
              </Box>
            ))}
          </Stack>

          {content.metadata && (
            <Box mt={4} pt={3} borderTopWidth={1}>
              <Text fontSize="sm" fontWeight="bold" mb={2} color="gray.600">Metadata:</Text>
              <SimpleGrid columns={{ base: 1, md: 2 }} spacing={3}>
                {Object.entries(content.metadata).map(([key, value]) => (
                  <Box key={key} p={2} bg="gray.50" borderRadius="md">
                    <Text fontSize="sm" fontWeight="bold" color="gray.600">{key}</Text>
                    <Text fontSize="sm">{typeof value === 'object' ? JSON.stringify(value, null, 2) : value}</Text>
                  </Box>
                ))}
              </SimpleGrid>
            </Box>
          )}
        </Card>
      );
    } else {
      return (
        <Card p={4} variant="outline" borderWidth={1} borderRadius="md" boxShadow="sm">
          <Stack spacing={3}>
            <Flex justify="space-between" align="center" mb={2}>
              <Box flex="1">
                <Box p={2} bg="gray.50" borderRadius="md">
                  <Text fontSize="sm" fontWeight="bold" color="blue.600" mb={1}>Instruction:</Text>
                  <Text whiteSpace="pre-wrap" fontSize="md">{content.instruction}</Text>
                </Box>
                {content.input && (
                  <Box p={2} bg="gray.50" borderRadius="md" mt={2}>
                    <Text fontSize="sm" fontWeight="bold" color="purple.600" mb={1}>Input:</Text>
                    <Text whiteSpace="pre-wrap" fontSize="md">{content.input}</Text>
                  </Box>
                )}
                <Box p={2} bg="gray.50" borderRadius="md" mt={2}>
                  <Text fontSize="sm" fontWeight="bold" color="green.600" mb={1}>Output:</Text>
                  <Text whiteSpace="pre-wrap" fontSize="md">{content.output}</Text>
                </Box>
              </Box>
              <IconButton
                icon={<FiEdit />}
                size="sm"
                variant="ghost"
                onClick={() => handleEditExample(example)}
                aria-label="Edit example"
              />
            </Flex>

            {content.metadata && (
              <Box mt={4} pt={3} borderTopWidth={1}>
                <Text fontSize="sm" fontWeight="bold" mb={2} color="gray.600">Metadata:</Text>
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={3}>
                  {Object.entries(content.metadata).map(([key, value]) => (
                    <Box key={key} p={2} bg="gray.50" borderRadius="md">
                      <Text fontSize="sm" fontWeight="bold" color="gray.600">{key}</Text>
                      <Text fontSize="sm">{typeof value === 'object' ? JSON.stringify(value, null, 2) : value}</Text>
                    </Box>
                  ))}
                </SimpleGrid>
              </Box>
            )}
          </Stack>
        </Card>
      );
    }
  };

  if (isLoadingDetails) {
    return (
      <Box textAlign="center" py={10}>
        <Spinner size="xl" />
        <Text mt={4}>Loading dataset...</Text>
      </Box>
    );
  }

  if (isErrorDetails) {
    return (
      <Box>
        <Button leftIcon={<FiArrowLeft />} onClick={() => navigate('/datasets')} mb={4}>
          Back to Datasets
        </Button>
        <Alert status="error">
          <AlertIcon />
          {detailsError?.response?.data?.detail || 'Failed to load dataset details'}
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Flex align="center" mb={6}>
        <Button leftIcon={<FiArrowLeft />} onClick={() => navigate('/datasets')} mr={2}>
          Back
        </Button>
        <Heading size="lg" flex="1">{dataset?.name}</Heading>
        <Button leftIcon={<FiDownload />} colorScheme="blue" mr={2} onClick={onExportOpen}>
          Export Dataset
        </Button>
        <IconButton icon={<FiEdit />} aria-label="Edit dataset" />
      </Flex>

      {dataset?.description && (
        <Text mb={6} color="gray.600">{dataset.description}</Text>
      )}

      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6}>
        <CardBody>
          <StatGroup>
            <Stat>
              <StatLabel>Examples</StatLabel>
              <StatNumber>{dataset?.example_count}</StatNumber>
            </Stat>

            <Stat>
              <StatLabel>Domain</StatLabel>
              <StatNumber fontSize="xl">{dataset?.domain}</StatNumber>
            </Stat>

            <Stat>
              <StatLabel>Format</StatLabel>
              <StatNumber fontSize="xl">
                <Badge colorScheme={dataset?.format === 'chat' ? 'blue' : 'green'} fontSize="md">
                  {dataset?.format}
                </Badge>
              </StatNumber>
            </Stat>

            <Stat>
              <StatLabel>Created</StatLabel>
              <StatNumber fontSize="xl">
                {new Date(dataset?.created_at).toLocaleDateString()}
              </StatNumber>
            </Stat>
          </StatGroup>
        </CardBody>
      </Card>

      <Flex justify="space-between" align="center" mb={4}>
        <Heading size="md">Examples</Heading>
        <HStack spacing={3}>
          <Button
            size="sm"
            variant={showQualityFilter ? 'solid' : 'outline'}
            colorScheme="purple"
            onClick={() => setShowQualityFilter(!showQualityFilter)}
            px={4}
          >
            Quality Filter {showQualityFilter ? '✓' : ''}
          </Button>
          <InputGroup maxW="300px">
            <InputLeftElement pointerEvents="none">
              <FiSearch color="gray.300" />
            </InputLeftElement>
            <Input
              placeholder="Search examples..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setPage(1);
              }}
            />
          </InputGroup>
        </HStack>
      </Flex>

      {showQualityFilter && (
        <Card p={4} mb={4} bg="purple.50" borderColor="purple.200" borderWidth={1}>
          <VStack align="stretch" spacing={3}>
            <Text fontWeight="bold" fontSize="sm">Filter by Quality Score</Text>
            <HStack spacing={4}>
              <Box flex={1}>
                <Text fontSize="xs" mb={2}>Min: {minQuality.toFixed(1)}</Text>
                <Slider
                  value={minQuality}
                  onChange={(val) => {
                    setMinQuality(val);
                    setPage(1);
                  }}
                  min={0}
                  max={10}
                  step={0.1}
                  colorScheme="purple"
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb />
                </Slider>
              </Box>
              <Box flex={1}>
                <Text fontSize="xs" mb={2}>Max: {maxQuality.toFixed(1)}</Text>
                <Slider
                  value={maxQuality}
                  onChange={(val) => {
                    setMaxQuality(val);
                    setPage(1);
                  }}
                  min={0}
                  max={10}
                  step={0.1}
                  colorScheme="purple"
                >
                  <SliderTrack>
                    <SliderFilledTrack />
                  </SliderTrack>
                  <SliderThumb />
                </Slider>
              </Box>
            </HStack>
            <HStack justify="space-between">
              <Text fontSize="xs" color="gray.600">
                Showing examples with quality between {minQuality.toFixed(1)} and {maxQuality.toFixed(1)}
              </Text>
              <Button
                size="xs"
                variant="ghost"
                onClick={() => {
                  setMinQuality(0);
                  setMaxQuality(10);
                  setPage(1);
                }}
              >
                Reset
              </Button>
            </HStack>
          </VStack>
        </Card>
      )}

      {searchQuery && examplesData && (
        <Text fontSize="sm" color="gray.600" mb={4}>
          Found {examplesData.total} {examplesData.total === 1 ? 'match' : 'matches'} for "{searchQuery}"
        </Text>
      )}

      {selectedExamples.length > 0 && (
        <Card p={4} mb={4} bg="blue.50" borderColor="blue.300" borderWidth={2}>
          <Flex justify="space-between" align="center">
            <Text fontWeight="bold">
              {selectedExamples.length} {selectedExamples.length === 1 ? 'example' : 'examples'} selected
            </Text>
            <HStack spacing={2}>
              <Button
                size="sm"
                leftIcon={<FiDownload />}
                onClick={handleBulkExport}
              >
                Export Selected
              </Button>
              <Button
                size="sm"
                colorScheme="red"
                leftIcon={<FiTrash2 />}
                onClick={handleBulkDelete}
              >
                Delete Selected
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setSelectedExamples([]);
                  setIsSelectAll(false);
                }}
              >
                Clear Selection
              </Button>
            </HStack>
          </Flex>
        </Card>
      )}

      {isLoadingExamples ? (
        <Box textAlign="center" py={4}>
          <Spinner />
          <Text mt={2}>Loading examples...</Text>
        </Box>
      ) : isErrorExamples ? (
        <Alert status="error">
          <AlertIcon />
          {examplesError?.response?.data?.detail || 'Failed to load examples'}
        </Alert>
      ) : (
        <>
          <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={4} overflow="hidden">
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th width="50px">
                    <Checkbox
                      isChecked={isSelectAll}
                      onChange={handleSelectAll}
                      colorScheme="blue"
                    />
                  </Th>
                  <Th width="80px">ID</Th>
                  <Th>Content</Th>
                </Tr>
              </Thead>
              <Tbody>
                {examplesData?.examples.map((example, index) => (
                  <Tr key={index}>
                    <Td>
                      <Checkbox
                        isChecked={selectedExamples.includes(example.id)}
                        onChange={() => handleSelectExample(example.id)}
                        colorScheme="blue"
                      />
                    </Td>
                    <Td isNumeric>
                      {((page - 1) * pageSize) + index + 1}
                    </Td>
                    <Td>{formatExample(example)}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </Card>

          {examplesData?.total_pages > 1 && (
            <Flex justify="center" mt={4}>
              <Button
                onClick={handlePreviousPage}
                isDisabled={page === 1}
                mr={2}
              >
                Previous
              </Button>
              <Text alignSelf="center" mx={4}>
                Page {page} of {examplesData.total_pages}
              </Text>
              <Button
                onClick={handleNextPage}
                isDisabled={page >= examplesData.total_pages}
                ml={2}
              >
                Next
              </Button>
            </Flex>
          )}
        </>
      )}

      {/* Export Wizard Modal */}
      <Modal isOpen={isExportOpen} onClose={onExportClose} size="4xl">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Export Dataset</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            <Text mb={6} color="gray.600">
              Choose the format that best fits your use case. Each format is optimized for different platforms and workflows.
            </Text>

            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
              {exportFormats.map((format) => (
                <Card
                  key={format.key}
                  p={4}
                  cursor="pointer"
                  borderWidth={2}
                  borderColor={selectedExportFormat?.key === format.key ? 'blue.500' : 'gray.200'}
                  _hover={{ borderColor: 'blue.300', shadow: 'md' }}
                  onClick={() => setSelectedExportFormat(format)}
                >
                  <VStack align="stretch" spacing={3}>
                    <HStack>
                      <Icon as={format.icon} boxSize={6} color="blue.500" />
                      <Heading size="md">{format.name}</Heading>
                    </HStack>

                    <Text fontSize="sm" color="gray.600">
                      {format.description}
                    </Text>

                    <Box>
                      <Text fontSize="xs" fontWeight="bold" mb={2} color="gray.500">
                        USE CASES:
                      </Text>
                      <List spacing={1}>
                        {format.useCases.map((useCase, idx) => (
                          <ListItem key={idx} fontSize="xs">
                            <ListIcon as={FiCheck} color="green.500" />
                            {useCase}
                          </ListItem>
                        ))}
                      </List>
                    </Box>

                    <Box>
                      <Text fontSize="xs" fontWeight="bold" mb={2} color="gray.500">
                        FEATURES:
                      </Text>
                      <List spacing={1}>
                        {format.features.map((feature, idx) => (
                          <ListItem key={idx} fontSize="xs" color="gray.600">
                            • {feature}
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  </VStack>
                </Card>
              ))}
            </SimpleGrid>

            {selectedExportFormat && (
              <Box mt={6} p={4} bg="blue.50" borderRadius="md">
                <Text fontWeight="bold" mb={2}>Preview ({selectedExportFormat.name} format):</Text>
                <Code p={4} borderRadius="md" display="block" whiteSpace="pre-wrap" fontSize="xs">
                  {selectedExportFormat.key === 'openai'
                    ? '{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}'
                    : selectedExportFormat.key === 'alpaca'
                      ? '{"instruction": "...", "input": "...", "output": "..."}'
                      : selectedExportFormat.key === 'sharegpt'
                        ? '{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}'
                        : selectedExportFormat.key === 'langchain'
                          ? '{"page_content": "...", "metadata": {...}}'
                          : '# HuggingFace Dataset\n\nSee README.md for details'}
                </Code>
              </Box>
            )}
          </ModalBody>

          <ModalFooter>
            <Button variant="ghost" mr={3} onClick={onExportClose}>
              Cancel
            </Button>
            <Button
              colorScheme="blue"
              onClick={() => selectedExportFormat && handleExport(selectedExportFormat)}
              isDisabled={!selectedExportFormat}
              leftIcon={<FiDownload />}
            >
              Export as {selectedExportFormat?.name || 'Format'}
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default DatasetDetail;
