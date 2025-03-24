import React, { useState } from 'react';
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
} from '@chakra-ui/react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FiArrowLeft, FiDownload, FiEdit, FiSave, FiX } from 'react-icons/fi';
import { getDatasetDetails, getDatasetExamples, updateExample } from '../services/api';

const DatasetDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [editingExample, setEditingExample] = useState(null);
  const [editContent, setEditContent] = useState(null);
  const toast = useToast();
  const queryClient = useQueryClient();
  
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
  
  // Fetch dataset examples
  const { 
    data: examplesData,
    isLoading: isLoadingExamples,
    isError: isErrorExamples,
    error: examplesError
  } = useQuery({
    queryKey: ['dataset-examples', id, page, pageSize],
    queryFn: () => getDatasetExamples(id, { page, page_size: pageSize }),
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
        <IconButton icon={<FiDownload />} aria-label="Download dataset" mr={2} onClick={downloadDataset} />
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
      
      <Heading size="md" mb={4}>Examples</Heading>
      
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
                  <Th width="80px">ID</Th>
                  <Th>Content</Th>
                </Tr>
              </Thead>
              <Tbody>
                {examplesData?.examples.map((example, index) => (
                  <Tr key={index}>
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
    </Box>
  );
};

export default DatasetDetail;
